"""Read-only forensic analysis for the completed recurring state-management run."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("results/model_version_aware_recurring_state_management_full_20260827T134710Z")
OUTPUT = Path("research/diagnostics/model_version_aware_recurring_state_management_20260827")
SEEDS = tuple(range(20260821, 20260831))
FULL = "full_reset"
ISOLATION = "version_indexed_learner_state_isolation"
POLICIES = (FULL, ISOLATION)
IGNORED = {"policy", "seed", "runtime_seconds", "timestamp"}


def mean(rows, field): return sum(float(row[field]) for row in rows) / len(rows)
def rate(rows, field): return sum(int(row[field]) for row in rows) / len(rows)
def phase(rows, identifier): return [row for row in rows if row["phase"] == identifier]


def metric(seed, phase_id, full, isolated):
    f, i = phase(full, phase_id), phase(isolated, phase_id)
    arm_matches = sum(a["selected_arm"] == b["selected_arm"] for a, b in zip(f, i))
    first_difference = next((n + 1 for n, (a, b) in enumerate(zip(f, i)) if a["selected_arm"] != b["selected_arm"]), None)
    return {
        "seed": seed, "phase": phase_id,
        "full_reward": mean(f, "reward"), "isolation_reward": mean(i, "reward"),
        "reward_difference": mean(i, "reward") - mean(f, "reward"),
        "recovery_difference": rate(i, "recovered") - rate(f, "recovered"),
        "accuracy_difference": rate(i, "defended_correct") - rate(f, "defended_correct"),
        "post_risk_difference": mean(i, "observed_post_risk") - mean(f, "observed_post_risk"),
        "same_selected_arm_episodes": arm_matches,
        "different_selected_arm_episodes": len(f) - arm_matches,
        "first_selected_arm_difference_episode": first_difference,
        "first_10_reward_difference": mean(i[:10], "reward") - mean(f[:10], "reward"),
        "last_90_reward_difference": mean(i[10:], "reward") - mean(f[10:], "reward"),
        "full_arm_counts": json.dumps(Counter(row["selected_arm"] for row in f), sort_keys=True),
        "isolation_arm_counts": json.dumps(Counter(row["selected_arm"] for row in i), sort_keys=True),
        "full_updates_before_D": int(f[0]["bandit_updates_before"]),
        "isolation_updates_before_D": int(i[0]["bandit_updates_before"]),
        "full_updates_after_D": int(f[-1]["bandit_updates_after"]),
        "isolation_updates_after_D": int(i[-1]["bandit_updates_after"]),
        "full_cumulative_reward_at_D_start": float(f[0]["cumulative_reward_after"]) - float(f[0]["reward"]),
        "isolation_cumulative_reward_at_D_start": float(i[0]["cumulative_reward_after"]) - float(i[0]["reward"]),
    }


def summarize(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "sample_standard_deviation": statistics.stdev(values), "minimum": min(values), "maximum": max(values),
            "positive_count": sum(value > 0 for value in values), "negative_count": sum(value < 0 for value in values),
            "tie_count": sum(value == 0 for value in values)}


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    required = [ROOT / "final_comparison.json", ROOT / "paired_phase_c.csv", ROOT / "paired_phase_d.csv", ROOT / "provenance.json", ROOT / "resolved_config.json", ROOT / "command_record.json"]
    if any(not path.exists() for path in required): raise FileNotFoundError([str(path) for path in required if not path.exists()])
    rows, audits = {}, {}
    for seed in SEEDS:
        for policy in POLICIES:
            directory = ROOT / f"seed_{seed}" / policy
            path = directory / "episodes.csv"
            audit = directory / "boundary_state_audit.json"
            if not path.exists() or not audit.exists(): raise FileNotFoundError(directory)
            with path.open(newline="", encoding="utf-8") as handle: rows[seed, policy] = list(csv.DictReader(handle))
            audits[seed, policy] = json.loads(audit.read_text(encoding="utf-8"))
    equality, c_rows, d_rows, restoration = [], [], [], []
    for seed in SEEDS:
        full, isolated = rows[seed, FULL], rows[seed, ISOLATION]
        phase_counts = {policy: dict(Counter(row["phase"] for row in rows[seed, policy])) for policy in POLICIES}
        a_b_exact = {}
        for phase_id in "AB":
            left = [{key: value for key, value in row.items() if key not in IGNORED} for row in phase(full, phase_id)]
            right = [{key: value for key, value in row.items() if key not in IGNORED} for row in phase(isolated, phase_id)]
            a_b_exact[phase_id] = left == right
        equality.append({"seed": seed, "phase_counts": json.dumps(phase_counts, sort_keys=True), **{f"phase_{key}_exact": value for key, value in a_b_exact.items()}})
        c_rows.append(metric(seed, "C", full, isolated)); d_rows.append(metric(seed, "D", full, isolated))
        iso_audit = audits[seed, ISOLATION]
        full_audit = audits[seed, FULL]
        b_to_c = next(item for item in iso_audit if item.get("boundary_before_phase") == "C")
        d = next(item for item in iso_audit if item.get("boundary_before_phase") == "D")
        full_d = next(item for item in full_audit if item.get("boundary_before_phase") == "D")
        restoration.append({"seed": seed,
            "B_state_hash_saved_at_B_to_C": b_to_c["departing_state_hash"],
            "D_active_state_hash": d["active_state_hash_at_boundary"],
            "B_hash_equals_D_active_hash": b_to_c["departing_state_hash"] == d["active_state_hash_at_boundary"],
            "D_mode": d["target_state_mode"], "D_restore_hash_check": d["restored_state_hash_matches_saved"],
            "D_registry_keys": json.dumps(d["registry_keys"]), "D_integrity": d["integrity_passed_at_boundary"],
            "D_full_reset_mode": full_d["target_state_mode"], "D_full_reset_prior_checks": json.dumps(full_d["fresh_state_prior_checks"], sort_keys=True),
            "D_full_reset_state_hash": full_d["active_state_hash_at_boundary"],
            "D_isolation_updates_before": int(phase(rows[seed, ISOLATION], "D")[0]["bandit_updates_before"]),
            "D_full_updates_before": int(phase(rows[seed, FULL], "D")[0]["bandit_updates_before"]),
        })
    d_reward = [row["reward_difference"] for row in d_rows]
    report = {"source_run": str(ROOT), "artifact_verification": {
        "required_root_files_present": True, "seed_count": len(SEEDS), "policy_count": len(POLICIES),
        "episode_csv_count": len(rows), "boundary_audit_count": len(audits),
        "all_phase_counts_100": all(all(counts == 100 for counts in json.loads(row["phase_counts"]).values().__iter__().__next__().values()) for row in equality),
        "raw_phase_A_B_exact": all(row["phase_A_exact"] and row["phase_B_exact"] for row in equality)},
        "phase_d_recomputed": {"reward_difference": summarize(d_reward),
            "recovery_difference_mean": statistics.mean(row["recovery_difference"] for row in d_rows),
            "accuracy_difference_mean": statistics.mean(row["accuracy_difference"] for row in d_rows),
            "post_risk_difference_mean": statistics.mean(row["post_risk_difference"] for row in d_rows)},
        "restoration": {"all_B_hashes_equal_D_active_hashes": all(row["B_hash_equals_D_active_hash"] for row in restoration),
            "all_D_restore_hash_checks": all(row["D_restore_hash_check"] for row in restoration),
            "all_D_isolation_modes_restored": all(row["D_mode"] == "restored_existing_version" for row in restoration),
            "all_D_full_reset_modes_fresh": all(row["D_full_reset_mode"] == "fresh_full_reset" for row in restoration)},
        "limitations": ["Artifacts retain full-state hashes and cache/prior audit booleans, but not numeric B/f matrices or serialized RNG/CUSUM payloads; their magnitudes cannot be independently quantified post hoc.", "CSV posterior fields are only selected-arm projections/uncertainty, not complete per-arm state."]}
    OUTPUT.mkdir(parents=True)
    for name, table in (("raw_equality_and_phase_counts.csv", equality), ("phase_c_raw_reconstruction.csv", c_rows), ("phase_d_raw_reconstruction.csv", d_rows), ("restoration_crosscheck.csv", restoration)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    (OUTPUT / "forensic_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text("# Read-only forensic outputs\n\nDerived from the completed recurring experiment; no source or result artifact was modified.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
