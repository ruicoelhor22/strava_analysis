from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from endurance_lab.prescription import prescribe_day


def backtest(
    start: date,
    end: date,
    database: str | Path | None = None,
) -> dict[str, Any]:
    if end < start:
        raise ValueError("Backtest end date must not precede start date")
    decisions = []
    cursor = start
    while cursor <= end:
        decisions.extend(item.to_dict() for item in prescribe_day(cursor, database, persist=False))
        cursor += timedelta(days=1)
    actions = Counter(item["action"] for item in decisions)
    reasons = Counter(rule for item in decisions for rule in item["rules_triggered"])
    changed = sum(action != "KEEP" for action in actions.elements())
    suspicious = []
    total = len(decisions)
    if total and changed / total > 0.5:
        suspicious.append(f"engine changed {changed / total:.0%} of planned workouts")
    rest = actions.get("REST", 0) + actions.get("SKIP", 0) + actions.get("REPLACE_WITH_EASY", 0)
    if total and rest / total > 0.25:
        suspicious.append(f"rest/skip/replacement actions account for {rest / total:.0%} of prescriptions")
    if actions.get("PROGRESS_SESSION", 0):
        suspicious.append("automatic progression occurred despite conservative default")
    key_changes = sum(
        item["action"] not in {"KEEP", "CONDITIONAL"}
        and str((item.get("decision_trace") or {}).get("planned_session", {}).get("priority", "")).startswith("A")
        for item in decisions
    )
    key_cancellations = sum(
        item["action"] in {"SKIP", "REST", "REPLACE_WITH_EASY"}
        and str((item.get("decision_trace") or {}).get("planned_session", {}).get("priority", "")).startswith("A")
        for item in decisions
    )
    if key_cancellations:
        suspicious.append(f"{key_cancellations} A-priority session(s) were cancelled or replaced")
    optional_upgrades = sum(
        str((item.get("decision_trace") or {}).get("planned_session", {}).get("priority", "")).startswith("C")
        and item["action"] == "PROGRESS"
        for item in decisions
    )
    if optional_upgrades:
        suspicious.append(f"{optional_upgrades} optional session(s) were upgraded or made more demanding")
    strength_cancellations = sum(
        item["action"] in {"SKIP", "REST"}
        and "lower_body_strength_interference" in item["rules_triggered"]
        for item in decisions
    )
    if strength_cancellations > 1:
        suspicious.append(f"strength interference caused {strength_cancellations} cancellations")
    return {
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "days_evaluated": (end - start).days + 1,
        "planned_sessions_evaluated": total,
        "decision_distribution": dict(actions),
        "rule_frequency": dict(reasons),
        "changed_sessions": changed,
        "key_sessions_changed": key_changes,
        "key_sessions_cancelled_or_replaced": key_cancellations,
        "optional_sessions_upgraded": optional_upgrades,
        "strength_related_cancellations": strength_cancellations,
        "suspicious_behavior": suspicious,
        "decisions": decisions,
    }
