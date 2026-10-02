"""
CRRA Lab C4 — Portfolio Orchestrator

Runs the whole renewal review as a LangGraph StateGraph:

    analysis  ->  policy_check  ->  [hitl]  ->  report

The HITL gate is the point of this lab. An agent may recommend, but under the
procurement policy it may not commit Zensar to anything on its own.

Prerequisites:
    1. python data/kb_setup.py                (loads the policy KB)
    2. python mcp_server/contract_shim.py     (second terminal, port 5001)

Run from the project root:
    python orchestrator/supervisor.py
"""

import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# Running as `python orchestrator/supervisor.py` puts only the orchestrator/
# folder on the import path, not the project root — so guardrails/ would not be
# found. Add the project root explicitly, before any project-local import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402
import chromadb  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger  # noqa: E402

# Box-drawing characters must not crash a console that can't print them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv(PROJECT_ROOT / ".env")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
KB_PATH = PROJECT_ROOT / "data" / "chroma_db"   # where data/kb_setup.py saves the KB
MAX_ROUNDS = 5
MAX_TOKENS = 4096   # room for adaptive thinking plus the tool call
RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

# Ignore corporate proxy settings for calls to our own localhost API
http = requests.Session()
http.trust_env = False

audit = AuditLogger()


def extract_text(response) -> str:
    """First text block — safe even when a thinking block comes first."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

class ContractState(TypedDict):
    contract_id: str
    contract: dict

    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int

    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str

    final_status: str


# ══════════════════════════════════════════════════════════════
# KB
# ══════════════════════════════════════════════════════════════

def _load_kb():
    """Open the KB that kb_setup.py saved to disk; build it here if it is missing."""
    kb_client = chromadb.PersistentClient(path=str(KB_PATH))
    try:
        collection = kb_client.get_collection(KB_COLLECTION)
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    from data.kb_setup import chunk_article  # reuse Lab C1's chunker

    try:
        kb_client.delete_collection(KB_COLLECTION)
    except Exception:
        pass
    # Cosine distance keeps (1 - distance) a true 0-1 confidence
    collection = kb_client.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted((PROJECT_ROOT / "data" / "kb").glob("*.md")):
        for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"  (built policy KB: {len(ids)} chunks)")
    return collection


KB = _load_kb()


def search_policy(query: str) -> list[dict]:
    # Each article has at most 4 sections, so 6 hits always span 2+ files
    res = KB.query(query_texts=[query], n_results=min(6, KB.count()))
    best: dict[str, tuple[float, str, str]] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta["source"]
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta["heading"], doc)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return [
        {"source": s, "section": h, "confidence": round(1 - d, 2), "text": t}
        for s, (d, h, t) in ranked
    ]


# ══════════════════════════════════════════════════════════════
# NODE 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════

ANALYSIS_TOOLS = [
    {
        "name": "search_policy",
        "description": "Search the procurement policy KB. Call at most twice.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {"type": "string"},
                "policy_citation": {"type": "string"},
                "estimated_annual_impact_inr": {"type": "integer"},
            },
            "required": ["recommendation", "confidence", "rationale", "policy_citation"],
        },
    },
]

ANALYSIS_SYSTEM = """You are the Renewal Analysis Agent for Zensar BizOps.

Decide one of RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE for the contract given.

Search the policy KB (at most twice) to find the governing rule, then call
submit_recommendation exactly once.

Guidance:
- Utilisation under 40% with an overlapping vendor in the same category -> CONSOLIDATE.
- An unassigned owner is not a licence to TERMINATE — nobody has confirmed the
  capability is unneeded, so recommend conservatively and say why.
- Proposed uplift above 15% -> RENEGOTIATE, never accept at first offer.
- Healthy utilisation plus modest uplift -> RENEW.

LOW confidence is a valid, useful answer. Say so plainly instead of inventing
certainty, and do not keep searching hoping for a cleaner picture."""


def analysis_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print(f"\n{'═' * 66}")
    print(f"CONTRACT: {cid}")
    print("═" * 66)
    print("\n▶ ANALYSIS AGENT")

    try:
        r = http.get(f"{CONTRACT_API}/api/contracts/{cid}", timeout=10)
        contract = r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Contract API unreachable: {type(e).__name__}")
        print("     Is mcp_server/contract_shim.py running on port 5001?")
        audit.log("AnalysisAgent", "contract_fetch_failed", cid,
                  f"Contract API unreachable at {CONTRACT_API}", "N/A")
        return {**state, "final_status": "ERROR_API_UNREACHABLE", "hitl_required": False}
    except ValueError:
        audit.log("AnalysisAgent", "contract_fetch_failed", cid,
                  f"Non-JSON reply (HTTP {r.status_code})", "N/A")
        return {**state, "final_status": "ERROR_BAD_RESPONSE", "hitl_required": False}

    if "error" in contract:
        print(f"  ✗ {contract['error']}")
        audit.log("AnalysisAgent", "contract_fetch_failed", cid, contract["error"], "N/A")
        return {**state, "final_status": "ERROR_NOT_FOUND", "hitl_required": False}

    util = contract["utilisation_pct"]
    print(
        f"  {contract['vendor']} · {contract['category']} · band {contract['approval_band']}\n"
        f"  INR {contract['annual_value_inr']:,}/yr · "
        f"util {f'{util:g}%' if util is not None else 'n/a (no seats)'} · "
        f"uplift {contract['proposed_uplift_pct']:g}% · {contract['notice_state']}"
    )
    audit.log("AnalysisAgent", "contract_fetched", cid,
              f"{contract['vendor']} band {contract['approval_band']} {contract['notice_state']}", "N/A")

    facts = json.dumps(
        {
            k: contract[k]
            for k in (
                "contract_id", "vendor", "category", "owner", "annual_value_inr",
                "approval_band", "renewal_date", "notice_state", "auto_renew",
                "seats_purchased", "seats_active", "utilisation_pct",
                "proposed_uplift_pct",
            )
        },
        indent=2,
    )

    messages = [{"role": "user", "content": f"Analyse this contract:\n{facts}"}]
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"  ⚠️  No recommendation after {MAX_ROUNDS} rounds — recording LOW confidence.")
            audit.log("AnalysisAgent", "analysis_incomplete", cid,
                      f"Stopped after {MAX_ROUNDS} rounds", "N/A")
            return {
                **state, "contract": contract, "recommendation": "RENEGOTIATE",
                "confidence": "LOW", "rationale": "Analysis did not converge; needs manual review.",
                "policy_citation": "n/a", "estimated_annual_impact_inr": 0,
            }

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "medium"},
                system=ANALYSIS_SYSTEM,
                tools=ANALYSIS_TOOLS,
                messages=messages,
            )
        except anthropic.AuthenticationError:
            audit.log("AnalysisAgent", "model_auth_failed", cid, "Anthropic API key rejected (401)", "N/A")
            raise  # no point trying the other contracts with a bad key; main() explains
        except anthropic.APIError as e:
            print(f"  ✗ Model API error: {str(e)[:150]}")
            audit.log("AnalysisAgent", "model_api_error", cid, str(e)[:200], "N/A")
            return {**state, "contract": contract, "final_status": "ERROR_MODEL_API",
                    "hitl_required": False}

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            print(f"  Agent finished without a recommendation: {text[:150]}")
            audit.log("AnalysisAgent", "analysis_incomplete", cid,
                      f"Stopped ({response.stop_reason}) without submitting", "N/A")
            return {
                **state, "contract": contract, "recommendation": "RENEGOTIATE",
                "confidence": "LOW", "rationale": text[:300] or "No recommendation produced.",
                "policy_citation": "n/a", "estimated_annual_impact_inr": 0,
            }

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "search_policy":
                query = block.input.get("query", "")
                hits = search_policy(query)
                tops = ", ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})" for h in hits)
                print(f'  → policy: "{query[:40]}" -> {tops}')
                audit.log("AnalysisAgent", "search_policy", cid, f'"{query[:60]}" -> {tops}', "N/A")
                result = {"results": hits}

            elif block.name == "submit_recommendation":
                a = block.input
                # The schema enums guide the model; make sure they actually held
                if (a.get("recommendation") not in RECOMMENDATIONS
                        or a.get("confidence") not in CONFIDENCE_LEVELS):
                    result = {"error": f"recommendation must be one of {RECOMMENDATIONS} and "
                                       f"confidence one of {CONFIDENCE_LEVELS}. Submit again."}
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": json.dumps(result), "is_error": True})
                    continue
                print(f"  → {a['recommendation']} (confidence {a['confidence']})")
                audit.log(
                    "AnalysisAgent", "recommendation", cid,
                    f"{a['recommendation']} / {a['confidence']} — {a.get('policy_citation', 'n/a')}", "N/A",
                )
                return {
                    **state,
                    "contract": contract,
                    "recommendation": a["recommendation"],
                    "confidence": a["confidence"],
                    "rationale": a["rationale"],
                    "policy_citation": a.get("policy_citation", "n/a"),
                    "estimated_annual_impact_inr": a.get("estimated_annual_impact_inr", 0),
                }
            else:
                result = {"error": f"unknown tool {block.name}"}

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )

        messages.append({"role": "user", "content": tool_results})


# ══════════════════════════════════════════════════════════════
# NODE 2 — POLICY CHECK  (decides whether a human must sign off)
# ══════════════════════════════════════════════════════════════

def policy_check_node(state: ContractState) -> ContractState:
    print("\n▶ POLICY CHECK")
    contract = state.get("contract", {})
    if not contract or state.get("final_status", "").startswith("ERROR"):
        print("  Skipped — no analysis to check")
        audit.log("PolicyCheck", "skipped", state["contract_id"],
                  f"Upstream error: {state.get('final_status') or 'no contract data'}", "N/A")
        return {**state, "hitl_required": False}

    reasons: list[str] = []

    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"approval band {band} requires a named human approver")

    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append("inside the notice window — leverage already lost")

    if state.get("recommendation") == "TERMINATE":
        reasons.append("all terminations require written owner confirmation")

    if state.get("confidence") == "LOW":
        reasons.append("analysis confidence is LOW")

    if str(contract.get("owner", "")).upper() == "UNASSIGNED":
        reasons.append("no business owner on record")

    hitl_required = bool(reasons)
    reason_text = "; ".join(reasons) if reasons else "no policy trigger"

    print(f"  HITL required: {hitl_required}")
    if reasons:
        for r in reasons:
            print(f"    · {r}")

    audit.log("PolicyCheck", "evaluate_triggers", state["contract_id"], reason_text, "N/A")

    return {**state, "hitl_required": hitl_required, "hitl_reason": reason_text}


# ══════════════════════════════════════════════════════════════
# NODE 3 — HITL GATE
# ══════════════════════════════════════════════════════════════

def hitl_node(state: ContractState) -> ContractState:
    print("\n▶ HUMAN APPROVAL GATE")
    print(f"  Contract : {state['contract_id']}  ({state['contract'].get('vendor')})")
    print(f"  Proposed : {state['recommendation']}  (confidence {state['confidence']})")
    print(f"  Because  : {state['hitl_reason']}")
    print(f"  Policy   : {state['policy_citation']}")
    print(f"\n  {state['rationale']}")

    audit.log(
        "HITLGate", "approval_request", state["contract_id"],
        f"{state['recommendation']} — {state['hitl_reason']}", "PENDING",
    )

    try:
        answer = input("\n  Approve this recommendation? [y/n]: ").strip().lower()
        approved = answer in ("y", "yes")
        approver = "unknown"
        if approved:
            approver = input("  Approver name: ").strip() or "unnamed approver"
    except EOFError:
        # No terminal to ask (e.g. run from a script): never approve by default
        print("\n  (no interactive input available — treating as rejected)")
        approved, approver = False, "unknown"

    audit.log(
        "HITLGate", "approval_decision", state["contract_id"],
        f"{state['recommendation']} {'approved' if approved else 'rejected'}",
        "APPROVED" if approved else "REJECTED",
        actor=approver if approved else "unknown",
    )

    print(f"  → {'APPROVED by ' + approver if approved else 'REJECTED — no action will be taken'}")

    return {**state, "hitl_approved": approved, "approver": approver}


# ══════════════════════════════════════════════════════════════
# NODE 4 — REPORT
# ══════════════════════════════════════════════════════════════

def report_node(state: ContractState) -> ContractState:
    print("\n▶ REPORT")

    if state.get("final_status", "").startswith("ERROR"):
        print(f"  Skipped — {state['final_status']}")
        audit.log("Reporting", "final_status", state["contract_id"],
                  f"{state['final_status']}: no recommendation, nothing actioned", "N/A")
        return state

    if not state.get("hitl_required"):
        final = f"{state['recommendation']}_AUTO"
        note = "Actioned without human approval (no policy trigger)."
    elif state.get("hitl_approved"):
        final = f"{state['recommendation']}_APPROVED"
        note = f"Approved by {state.get('approver')}. Cleared to action."
    else:
        final = "ON_HOLD_REJECTED"
        note = "Rejected at the approval gate. No commitment made to the vendor."

    audit.log("Reporting", "final_status", state["contract_id"], f"{final}: {note}",
              "APPROVED" if state.get("hitl_approved")
              else ("REJECTED" if state.get("hitl_required") else "AUTO"),
              actor=state.get("approver") if state.get("hitl_approved") else "system")

    print(f"  FINAL STATUS: {final}")
    print(f"  {note}")

    return {**state, "final_status": final}


# ══════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════

def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)

    g.set_entry_point("analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges(
        "policy_check", route_after_policy_check, {"hitl": "hitl", "report": "report"}
    )
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.template to .env and add your key.")

    graph = build_graph()

    # CTR-1010 has no policy trigger and should skip the gate entirely — that
    # contrast is the point. The other two must stop for a human.
    portfolio = ["CTR-1010", "CTR-1012", "CTR-1006"]

    outcomes = []
    for cid in portfolio:
        state: ContractState = {
            "contract_id": cid, "contract": {}, "recommendation": "", "confidence": "",
            "rationale": "", "policy_citation": "", "estimated_annual_impact_inr": 0,
            "hitl_required": False, "hitl_reason": "", "hitl_approved": False,
            "approver": "", "final_status": "",
        }
        try:
            outcomes.append(graph.invoke(state))
        except anthropic.AuthenticationError:
            raise SystemExit(
                "\nAnthropic rejected the API key (401). Stopping the review.\n"
                "Check ANTHROPIC_API_KEY in .env (Console keys start with 'sk-ant-api03-'),\n"
                "and that no older key is set in this terminal: Remove-Item Env:ANTHROPIC_API_KEY"
            )

    print(f"\n\n{'═' * 66}")
    print("PORTFOLIO REVIEW COMPLETE")
    print("═" * 66)
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<7}{'Gate':<9}{'Final'}")
    print("-" * 66)
    for o in outcomes:
        gate = "HUMAN" if o.get("hitl_required") else "auto"
        print(
            f"{o['contract_id']:<11}{o.get('recommendation', '—'):<14}"
            f"{o.get('confidence', '—'):<7}{gate:<9}{o.get('final_status', '—')}"
        )

    audit.summary()


if __name__ == "__main__":
    main()
