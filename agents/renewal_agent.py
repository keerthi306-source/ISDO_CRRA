"""
CRRA Lab C3 — Renewal Analysis Agent

Reads one contract from the mock contract API, pulls the relevant procurement
policy from the ChromaDB knowledge base, checks for vendor overlap, and submits
a structured RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE recommendation.

Before running, from the project root:
    python data/kb_setup.py               # builds the policy KB on disk (once)
    python mcp_server/contract_shim.py    # contract API on :5001 (leave running)
Then, in a second terminal:
    python agents/renewal_agent.py

Needs ANTHROPIC_API_KEY, either set in the shell or in the project's .env file.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5"
MAX_ROUNDS = 5          # hard cap on model calls per contract, so the loop always ends
MAX_TOKENS = 8000
API_BASE = "http://localhost:5001"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "crra_policy"

TEST_CONTRACTS = ["CTR-1003", "CTR-1004", "CTR-1005", "CTR-1006", "CTR-1012"]
RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps procurement.
Today is 2025-04-01 (simulated). You analyse one contract and recommend what to do at renewal.

How to work:
1. Call get_contract for the contract you are given.
2. Call search_policy with short, specific queries for each issue the contract raises:
   notice window state, price uplift, low utilisation, missing owner, approval band.
3. If utilisation is below 60 percent, or the category is one that often overlaps,
   call find_category_overlap for its category.
4. Finish by calling submit_recommendation exactly once.
You have at most 5 turns in total, so request independent tools together in one turn.

Decision guidance:
- Utilisation below 40% with a viable overlapping vendor points to CONSOLIDATE.
- Utilisation below 20% with no business owner is not an automatic TERMINATE: an absent
  owner means escalate to a human, because nobody has confirmed the capability is unneeded.
- Proposed uplift above 15% is never accepted at first offer; that is RENEGOTIATE.
- High utilisation with modest uplift is a healthy RENEW.
- TERMINATE only where the capability itself is no longer required.

Rules for the recommendation:
- Ground every claim in the contract data and the policy text you retrieved. Do not rely
  on general procurement knowledge where the policy is silent; say it is silent instead.
- policy_citation must name the source file and section heading you relied on, exactly as
  search_policy returned them, e.g. "renegotiation_levers.md / Price uplift benchmarks".
- search_policy confidence below 0.40 is a weak match. Never cite a weak match as if it
  settles the question.
- human_approval_required must be true for any Band B or Band C contract, anything
  inside its notice window, any TERMINATE, and whenever your confidence is LOW. If the contract is INSIDE_WINDOW and in
  Band B or C, say in the rationale that it needs immediate human escalation.
- estimated_annual_impact_inr: NEGATIVE for a saving, positive for a cost, relative to
  renewing on the quoted terms. Use 0 for RENEW at existing terms or when there is no
  defensible estimate. State how you
  got the number in the rationale. A consolidation saving is always lower than the full
  value of the contract being dropped, because the receiving contract needs extra seats.
- If you are inside the notice window, say so plainly and explain how it weakens the
  negotiating position rather than hiding it.

Confidence — read this carefully:
- HIGH: the data is clear and a retrieved policy section directly covers this situation.
- MEDIUM: the direction is clear but something material is estimated or partly covered.
- LOW is a valid and useful answer, not a failure. Use it when the data conflicts, the
  owner is unassigned, the best policy match is weak, a tool returned an error, or a key
  fact is unknown (migration cost, data extraction terms, whether the capability is still
  needed). A LOW recommendation that is flagged for human review is far more useful to the
  approver than a confident guess. Never raise your confidence to sound decisive.
- If a tool fails, do not invent the missing data. Submit with LOW confidence, choose the
  least disruptive option, and explain in the rationale what could not be checked.
"""

TOOLS = [
    {
        "name": "get_contract",
        "description": "Fetch one contract from the contract management system, including "
                       "derived fields: notice_deadline, notice_state, utilisation_pct, approval_band.",
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the procurement policy knowledge base. Returns the best-matching "
                       "section from each of the top 2 policy files, with a confidence score "
                       "from 0 to 1. Below 0.40 is a weak match.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A short, specific question."}},
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": "List every vendor in a category with value and utilisation, plus the "
                       "category's total annual value. Use it to spot overlapping tools.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Submit the final recommendation. Call exactly once, at the end.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string", "enum": RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                "rationale": {"type": "string", "description": "Two to five sentences."},
                "policy_citation": {"type": "string", "description": "file / section heading"},
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Rough rupee impact. 0 for RENEW at existing terms. Negative means saving.",
                },
                "human_approval_required": {"type": "boolean"},
            },
            "required": [
                "contract_id", "recommendation", "confidence", "rationale",
                "policy_citation", "estimated_annual_impact_inr", "human_approval_required",
            ],
        },
    },
]


# ---------------------------------------------------------------- tools ----

def _api_get(path: str) -> dict:
    """GET from the contract API. Returns an error dict instead of raising."""
    try:
        # trust_env=False keeps corporate proxy settings from hijacking localhost calls
        r = httpx.get(f"{API_BASE}{path}", timeout=10, trust_env=False)
    except httpx.HTTPError as e:
        return {"error": f"Contract API unreachable at {API_BASE} ({type(e).__name__}). "
                         "Start it with: python mcp_server/contract_shim.py"}
    try:
        body = r.json()
    except ValueError:
        return {"error": f"Contract API returned non-JSON (HTTP {r.status_code})."}
    if r.status_code != 200:
        return {"error": body.get("error", f"HTTP {r.status_code}")}
    return body


def get_contract(contract_id: str) -> dict:
    return _api_get(f"/api/contracts/{contract_id}")


_policy_collection = None


def _open_or_build_kb():
    """Open the KB that kb_setup.py saved; if it is missing, build it here instead."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0:
            return col
    except Exception:
        pass
    print("   (policy KB not found - building it now from data/kb/ ...)")
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.kb_setup import chunk_article  # reuse Lab C1's chunker
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted((PROJECT_ROOT / "data" / "kb").glob("*.md")):
        for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
            ids.append(c["id"]); docs.append(c["document"]); metas.append(c["metadata"])
    col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def search_policy(query: str) -> dict:
    global _policy_collection
    try:
        if _policy_collection is None:
            _policy_collection = _open_or_build_kb()
        res = _policy_collection.query(
            query_texts=[query], n_results=min(10, _policy_collection.count())
        )
    except Exception as e:
        return {"error": f"Policy knowledge base unavailable ({e}). "
                         "Build it with: python data/kb_setup.py"}

    best_per_file: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta["source"]
        if src not in best_per_file or dist < best_per_file[src]["distance"]:
            best_per_file[src] = {"source": src, "section": meta["heading"],
                                  "distance": dist, "text": doc}
    top = sorted(best_per_file.values(), key=lambda h: h["distance"])[:2]
    return {
        "query": query,
        "results": [
            {"source": h["source"], "section": h["section"],
             "confidence": round(1 - h["distance"], 2), "text": h["text"]}
            for h in top
        ],
    }


def find_category_overlap(category: str) -> dict:
    data = _api_get("/api/categories")
    if "error" in data:
        return data
    for entry in data["categories"]:
        if entry["category"].lower() == category.strip().lower():
            return entry
    return {"error": f"No category named '{category}'.",
            "available_categories": [c["category"] for c in data["categories"]]}


def validate_recommendation(rec: dict) -> list[str]:
    """The schema enums guide the model; this check makes sure they held."""
    problems = []
    if rec.get("recommendation") not in RECOMMENDATIONS:
        problems.append(f"recommendation must be one of {RECOMMENDATIONS}")
    if rec.get("confidence") not in CONFIDENCE_LEVELS:
        problems.append(f"confidence must be one of {CONFIDENCE_LEVELS}")
    if not isinstance(rec.get("human_approval_required"), bool):
        problems.append("human_approval_required must be true or false")
    if not isinstance(rec.get("estimated_annual_impact_inr"), int):
        problems.append("estimated_annual_impact_inr must be a whole number")
    return problems


def enforce_approval_policy(rec: dict, contract: dict | None) -> dict:
    """Policy guardrail in code, not just in the prompt: Band B/C, TERMINATE, LOW
    confidence, or an unverified contract always need a human."""
    band = (contract or {}).get("approval_band")
    must_escalate = (
        contract is None or band in ("B", "C")
        or contract.get("notice_state") == "INSIDE_WINDOW"
        or rec["recommendation"] == "TERMINATE" or rec["confidence"] == "LOW"
    )
    if must_escalate and not rec["human_approval_required"]:
        return {**rec, "human_approval_required": True, "approval_overridden": True}
    return rec


def run_tool(name: str, args: dict, state: dict) -> tuple[dict, bool]:
    """Execute one tool call. Returns (result, is_error)."""
    if name == "get_contract":
        result = get_contract(args.get("contract_id", ""))
        if "error" not in result:
            state["contract"] = result
    elif name == "search_policy":
        result = search_policy(args.get("query", ""))
    elif name == "find_category_overlap":
        result = find_category_overlap(args.get("category", ""))
    elif name == "submit_recommendation":
        problems = validate_recommendation(args)
        if problems:
            return {"error": "Recommendation rejected: " + "; ".join(problems)}, True
        state["recommendation"] = enforce_approval_policy(dict(args), state["contract"])
        result = {"status": "recorded"}
    else:
        result = {"error": f"Unknown tool {name}"}
    return result, "error" in result


# ---------------------------------------------------------------- agent ----

def first_text(response) -> str:
    """Text of the first block that has any. A thinking block may come first."""
    return next((b.text for b in response.content if hasattr(b, "text")), "")


def analyse_contract(client: anthropic.Anthropic, contract_id: str) -> dict:
    state = {"contract": None, "recommendation": None}
    messages = [{"role": "user",
                 "content": f"Analyse contract {contract_id} and submit your recommendation."}]
    last_text, stop_note, rounds = "", "", 0

    while rounds < MAX_ROUNDS:
        rounds += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
        )
        last_text = first_text(response) or last_text
        # Send the full content back, thinking blocks included, so tool use stays valid
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            stop_note = f"model stopped ({response.stop_reason}) without submitting"
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"   round {rounds}: {block.name}({json.dumps(block.input)[:70]})")
            result, is_error = run_tool(block.name, block.input, state)
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result), "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})

        if state["recommendation"]:
            break
    else:
        stop_note = f"hit MAX_ROUNDS={MAX_ROUNDS} without submitting"

    contract = state["contract"] or {}
    summary = {"contract_id": contract_id, "vendor": contract.get("vendor", "?"),
               "band": contract.get("approval_band", "?"),
               "notice_state": contract.get("notice_state", "?"), "rounds": rounds}
    if state["recommendation"]:
        return {**summary, **state["recommendation"]}
    return {**summary, "recommendation": "NO DECISION", "confidence": "-",
            "human_approval_required": True, "estimated_annual_impact_inr": 0,
            "policy_citation": "-", "rationale": f"{stop_note}. Last model text: {last_text[:200]}"}


# ----------------------------------------------------------------- main ----

def load_dotenv_key() -> None:
    """Pick up ANTHROPIC_API_KEY from the project .env if the shell does not set it."""
    env_file = PROJECT_ROOT / ".env"
    if os.environ.get("ANTHROPIC_API_KEY") or not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"').strip("'")


def print_summary(results: list[dict]) -> None:
    header = (f"{'Contract':<10}{'Vendor':<20}{'Band':<6}{'Notice':<15}"
              f"{'Recommendation':<16}{'Conf':<8}{'Human?':<8}{'Impact INR':>13}{'Rounds':>8}")
    print("\n" + "=" * len(header))
    print("RENEWAL ANALYSIS SUMMARY  (simulated today 2025-04-01)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        human = ("YES*" if r.get("approval_overridden") else "YES") if r["human_approval_required"] else "no"
        print(f"{r['contract_id']:<10}{r['vendor'][:19]:<20}{r['band']:<6}{r['notice_state']:<15}"
              f"{r['recommendation']:<16}{r['confidence']:<8}{human:<8}"
              f"{r['estimated_annual_impact_inr']:>13,}{r['rounds']:>8}")
    print("-" * len(header))
    if any(r.get("approval_overridden") for r in results):
        print("* model said no human approval was needed; overridden by policy guardrail")
    print("\nCitations:")
    for r in results:
        print(f"  {r['contract_id']}: {r['policy_citation']}")


def main() -> None:
    load_dotenv_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (shell or .env). Set it and re-run.")

    health = _api_get("/health")
    if "error" in health:
        print(f"WARNING: {health['error']}\n  The agent will run, but get_contract will fail.\n")

    client = anthropic.Anthropic()
    results = []
    for cid in TEST_CONTRACTS:
        print(f"\n=== {cid} ===")
        try:
            r = analyse_contract(client, cid)
        except anthropic.APIError as e:
            r = {"contract_id": cid, "vendor": "?", "band": "?", "notice_state": "?",
                 "recommendation": "API ERROR", "confidence": "-",
                 "human_approval_required": True, "estimated_annual_impact_inr": 0,
                 "policy_citation": "-", "rationale": str(e)[:200], "rounds": 0}
        print(f"   -> {r['recommendation']} ({r['confidence']}): {r['rationale']}")
        results.append(r)
    print_summary(results)


if __name__ == "__main__":
    main()
