"""
CRRA Lab C4 — Audit Trail

Every agent decision and every human approval is written here. For a BizOps
process that commits real money, "the system decided" is not an acceptable
answer during an audit — you need to be able to show what was recommended, on
what evidence, and who signed it off.

The log is APPEND-ONLY by design. Re-running the orchestrator adds to the file
rather than replacing it, which is what an audit trail is supposed to do. If you
want a clean file for a demo, delete logs/audit_trail.jsonl by hand first.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def log(
        self,
        agent: str,
        action: str,
        contract_id: str,
        rationale: str = "",
        approval_status: str = "N/A",
        actor: str = "system",
    ) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "contract_id": contract_id,
            "rationale": rationale,
            "approval_status": approval_status,
            "actor": actor,
        }
        self.entries.append(entry)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        short = rationale[:70] + ("..." if len(rationale) > 70 else "")
        print(f"  [AUDIT] {agent}: {action} — {short}")
        return entry

    def summary(self) -> None:
        print(f"\n{'═' * 66}")
        print(f"AUDIT TRAIL — {len(self.entries)} entries this run")
        print("═" * 66)
        print(f"{'Time':<10}{'Agent':<22}{'Action':<24}{'Approval'}")
        print("-" * 66)
        for e in self.entries:
            t = e["timestamp"].split("T")[1]
            print(f"{t:<10}{e['agent']:<22}{e['action'][:22]:<24}{e['approval_status']}")
        print("-" * 66)
        print(f"Written to: {self.log_path}")
