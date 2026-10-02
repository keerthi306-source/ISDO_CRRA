"""CRRA Lab C2 — Mock Contract API for Lab C3's Analysis Agent.
Run from the project root:  python mcp_server/contract_shim.py  (port 5001)
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

CSV_PATH = Path(__file__).parent.parent / "data" / "contracts.csv"
# Fixed "today" so every participant sees identical results on any day
SIMULATED_TODAY = datetime(2025, 4, 1)
EDITABLE = ("status", "owner", "proposed_uplift_pct")
CONTRACTS: list[dict] = []


def enrich(row: dict) -> dict:
    """Type-convert one CSV row and add the four derived fields."""
    for key in ("annual_value_inr", "notice_days", "seats_purchased", "seats_active"):
        row[key] = int(row[key] or 0)
    row["proposed_uplift_pct"] = float(row.get("proposed_uplift_pct") or 0)
    if "auto_renew" in row:
        row["auto_renew"] = row["auto_renew"].strip().upper() in ("Y", "YES", "TRUE")

    renewal = datetime.strptime(row["renewal_date"], "%Y-%m-%d")
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.strftime("%Y-%m-%d")
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days

    if renewal < SIMULATED_TODAY:
        row["notice_state"] = "EXPIRED"
    elif deadline <= SIMULATED_TODAY:
        row["notice_state"] = "INSIDE_WINDOW"  # deadline passed, renewal not yet
    elif row["days_to_notice_deadline"] <= 30:
        row["notice_state"] = "APPROACHING"
    else:
        row["notice_state"] = "OPEN"

    # Seat-based contracts only; AMC/support contracts have no seats
    bought = row["seats_purchased"]
    row["utilisation_pct"] = round(100 * row["seats_active"] / bought, 1) if bought else None

    v = row["annual_value_inr"]
    row["approval_band"] = "A" if v < 1_000_000 else ("B" if v <= 5_000_000 else "C")
    return row


def load_contracts() -> None:
    """Read the CSV once and keep it in memory."""
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        CONTRACTS[:] = [enrich(r) for r in csv.DictReader(f)]


def find(contract_id: str) -> dict | None:
    return next((c for c in CONTRACTS if c["contract_id"].upper() == contract_id.upper()), None)


def not_found(contract_id: str):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "contracts_loaded": len(CONTRACTS),
                    "simulated_today": f"{SIMULATED_TODAY:%Y-%m-%d}"})


@app.get("/api/contracts")
def list_contracts():
    """Optional filters: ?category= ?band= ?notice_state= (case-insensitive)."""
    filters = {
        "category": request.args.get("category"),
        "approval_band": request.args.get("band"),
        "notice_state": request.args.get("notice_state"),
    }
    results = [
        c for c in CONTRACTS
        if all(not want or str(c[field]).lower() == want.lower()
               for field, want in filters.items())
    ]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    """Contracts renewing within ?days= (default 90), soonest first."""
    try:
        window = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be a whole number"}), 400
    results = sorted(
        (c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= window),
        key=lambda c: c["days_to_renewal"],
    )
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    c = find(contract_id)
    return jsonify(c) if c else not_found(contract_id)


@app.get("/api/categories")
def categories():
    """Vendors grouped by category, with total annual value per category."""
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append({
            k: c[k] for k in ("contract_id", "vendor", "annual_value_inr", "utilisation_pct")
        })
    summary = [
        {"category": cat, "vendor_count": len(items),
         "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
         "vendors": items}
        for cat, items in sorted(grouped.items())
    ]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """Update status, owner or proposed_uplift_pct. In memory only."""
    c = find(contract_id)
    if not c:
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    updates = {k: payload[k] for k in EDITABLE if k in payload}
    if not updates:
        return jsonify({"error": f"Send at least one of: {', '.join(EDITABLE)}"}), 400
    if "proposed_uplift_pct" in updates:
        try:
            updates["proposed_uplift_pct"] = float(updates["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return jsonify({"error": "proposed_uplift_pct must be a number"}), 400
    c.update(updates)
    return jsonify({"updated": sorted(updates), "contract": c})


load_contracts()  # at import, so `flask run` works too

if __name__ == "__main__":
    print(f"Mock Contract API: {len(CONTRACTS)} contracts, today={SIMULATED_TODAY:%Y-%m-%d}")
    print("http://localhost:5001/health")
    app.run(port=5001, debug=False)
