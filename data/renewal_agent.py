"""
CRRA Lab C3 — Renewal Analysis Agent

Given a contract, decides RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE, with a
confidence level and a citation to the procurement policy that justifies it.

Prerequisites:
    1. python data/kb_setup.py                  (Lab C1 — loads the policy KB)
    2. python mcp_server/contract_shim.py       (Lab C2 — in a second terminal)

Run from the project root:
    python agents/renewal_agent.py
"""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
import chromadb
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"

# Hard cap so a model that will not converge cannot loop forever
MAX_ROUNDS = 5


# ══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_text(response) -> str:
    """Return the first text block from a response.

    Newer models may put a thinking block first, so response.content[0].text is
    not safe — it raises AttributeError the moment adaptive thinking kicks in.
    Scan for the first block that actually has text instead.
    """
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  (built in Lab C1)
# ══════════════════════════════════════════════════════════════

def _load_kb():
    kb_client = chromadb.Client()
    try:
        return kb_client.get_collection(KB_COLLECTION)
    except Exception:
        print("  KB collection not found — building it now from data/kb/ ...")
        from data.kb_setup import chunk_article  # reuse Lab C1's chunker

        collection = kb_client.create_collection(KB_COLLECTION)
        ids, docs, metas = [], [], []
        for md in sorted((Path(__file__).parent.parent / "data" / "kb").glob("*.md")):
            for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
                ids.append(c["id"])
                docs.append(c["document"])
                metas.append(c["metadata"])
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  Built KB with {len(ids)} chunks.")
        return collection


KB = _load_kb()


# ══════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_contract",
        "description": (
            "Fetch one contract by its ID, including derived fields: approval_band, "
            "notice_state, utilisation_pct, days_to_renewal, proposed_uplift_pct."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string", "description": "e.g. CTR-1004"}
            },
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy knowledge base. Use this to find "
            "the rule that applies before making a recommendation. Call this at "
            "most twice per contract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain-English policy question, e.g. 'who approves a 60 lakh contract'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": (
            "List every other active contract in the same category, with vendor, "
            "value and utilisation. Use this to judge whether consolidation is viable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "e.g. Observability"}
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation for this contract. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {
                    "type": "string",
                    "description": "Two or three sentences. State the specific numbers that drove the decision.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "The policy file and section that supports this, e.g. 'auto_renewal_rules.md §Escalation trigger'",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Rough rupee impact. 0 for RENEW at existing terms. Negative means saving.",
                },
                "human_approval_required": {
                    "type": "boolean",
                    "description": "True if policy requires a named human to approve before acting.",
                },
            },
            "required": [
                "contract_id",
                "recommendation",
                "confidence",
                "rationale",
                "policy_citation",
                "human_approval_required",
            ],
        },
    },
]


def tool_get_contract(contract_id: str) -> dict:
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{contract_id}", timeout=10)
        if r.status_code == 404:
            return {"error": f"{contract_id} not found"}
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Contract API unreachable ({e}). Is contract_shim.py running on port 5001?"}


def tool_search_policy(query: str) -> dict:
    """Return whole policy sections, grouped by source article.

    Returning arbitrary top-N chunks scattered across files gives the model
    fragments with no context. Grouping by source and returning the best section
    per article keeps each result readable and citable.
    """
    res = KB.query(query_texts=[query], n_results=6)

    best_per_source: dict[str, tuple[float, str, str]] = {}
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        src = meta["source"]
        if src not in best_per_source or dist < best_per_source[src][0]:
            best_per_source[src] = (dist, meta["heading"], doc)

    ranked = sorted(best_per_source.items(), key=lambda kv: kv[1][0])[:2]
    return {
        "results": [
            {
                "source": src,
                "section": heading,
                "confidence": round(1 - dist, 2),
                "text": doc,
            }
            for src, (dist, heading, doc) in ranked
        ]
    }


def tool_find_category_overlap(category: str) -> dict:
    try:
        r = requests.get(f"{CONTRACT_API}/api/categories", timeout=10)
        for c in r.json()["categories"]:
            if c["category"].lower() == category.lower():
                return c
        return {"category": category, "vendor_count": 0, "vendors": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"Contract API unreachable ({e})."}


SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps.

For the contract you are given, decide one of: RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE.

Method — follow in order:
1. Call get_contract to read the real numbers. Never assume them.
2. Call search_policy to find the governing rule. Call it at most twice.
3. Call find_category_overlap ONLY if you are considering CONSOLIDATE.
4. Call submit_recommendation exactly once to finish.

Decision guidance:
- Utilisation below 40% with a viable overlapping vendor points to CONSOLIDATE.
- Utilisation below 20% with no business owner is not an automatic TERMINATE — an
  absent owner means escalate to a human, because nobody has confirmed the
  capability is unneeded.
- Proposed uplift above 15% is never accepted at first offer; that is RENEGOTIATE.
- High utilisation with modest uplift is a healthy RENEW.
- TERMINATE only where the capability itself is no longer required.

Confidence:
- HIGH means the numbers and the policy point the same way with no ambiguity.
- MEDIUM means the recommendation is sound but rests on an assumption you should name.
- LOW means genuinely unclear. LOW is a perfectly valid answer — say so rather than
  inventing certainty. Do not keep calling tools hoping for a cleaner picture.

Set human_approval_required to true whenever policy demands it (approval Bands B and C,
anything inside its notice window, and every termination).

Always cite the specific policy file and section that justifies your call."""


def analyse_contract(contract_id: str) -> dict | None:
    print(f"\n{'═' * 62}")
    print(f"ANALYSING: {contract_id}")
    print("═" * 62)

    messages = [
        {"role": "user", "content": f"Analyse contract {contract_id} and recommend an action."}
    ]
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"  ⚠️  Stopped after {MAX_ROUNDS} rounds with no recommendation — treating as LOW confidence.")
            return None

        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            if text:
                print(f"  Agent said: {text[:200]}")
            return None

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            name, args = block.name, block.input

            if name == "get_contract":
                result = tool_get_contract(args["contract_id"])
                if "error" in result:
                    print(f"  ✗ {result['error']}")
                else:
                    print(
                        f"  → contract: band {result['approval_band']}  "
                        f"{result['notice_state']}  util {result['utilisation_pct']}%  "
                        f"uplift {result['proposed_uplift_pct']}%  "
                        f"INR {result['annual_value_inr']:,}"
                    )

            elif name == "search_policy":
                result = tool_search_policy(args["query"])
                tops = ", ".join(
                    f"{r['source']} §{r['section']} ({r['confidence']:.0%})"
                    for r in result["results"]
                )
                print(f'  → policy "{args["query"][:44]}" -> {tops}')

            elif name == "find_category_overlap":
                result = tool_find_category_overlap(args["category"])
                names = ", ".join(
                    f"{v['vendor']} ({v['utilisation_pct']}%)" for v in result.get("vendors", [])
                )
                print(f"  → overlap in {args['category']}: {names or 'none'}")

            elif name == "submit_recommendation":
                rec = args
                print(f"\n  ┌─ RECOMMENDATION ─────────────────────────────────")
                print(f"  │ {rec['recommendation']}   confidence {rec['confidence']}")
                print(f"  │ Policy: {rec['policy_citation']}")
                impact = rec.get("estimated_annual_impact_inr")
                if impact:
                    print(f"  │ Impact: INR {impact:,}/yr")
                print(f"  │ Human approval required: {rec['human_approval_required']}")
                print(f"  └──────────────────────────────────────────────────")
                print(f"  {rec['rationale']}")
                return rec

            else:
                result = {"error": f"unknown tool {name}"}

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key.")

    # A deliberate spread: healthy renewal, high uplift, low utilisation with an
    # overlap, an orphaned contract, and a high-value one inside its notice window.
    test_contracts = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]

    results = []
    for cid in test_contracts:
        rec = analyse_contract(cid)
        if rec:
            results.append(rec)

    print(f"\n\n{'═' * 62}")
    print("PORTFOLIO SUMMARY")
    print("═" * 62)
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Approval?':<11}Impact")
    print("-" * 62)
    for r in results:
        impact = r.get("estimated_annual_impact_inr") or 0
        impact_s = f"INR {impact:,}" if impact else "—"
        print(
            f"{r['contract_id']:<11}{r['recommendation']:<14}{r['confidence']:<8}"
            f"{str(r['human_approval_required']):<11}{impact_s}"
        )
    needs_human = sum(1 for r in results if r["human_approval_required"])
    print("-" * 62)
    print(f"{len(results)} analysed · {needs_human} require human approval before action")


if __name__ == "__main__":
    main()
