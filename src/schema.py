"""
Feature schema for the anchored_globem_hrv batch.

The raw file mixes three "tiers" under one flat record shape:
  tier 1 (n=480, source="globem")  -- phone passive-sensing + Fitbit-derived sleep
  tier 3 (n=78,  source="globem")  -- same schema as tier 1, disjoint subjects, undocumented origin
  tier 2 (n=40,  source="hrv")     -- wearable HRV + diary sleep metrics, different units/granularity

tier 1 and tier 3 share one feature namespace ("globem"); tier 2 has its own
("hrv"). The two namespaces do NOT share units (e.g. globem sleep_duration_min
is minutes, hrv sleep_duration is hours) so they are kept as separate, prefixed
columns rather than silently merged -- merging them is a real modeling
decision for the team, not something to default silently in a baseline.
"""

# Raw per-day numeric features present for globem rows (tier 1 + tier 3).
GLOBEM_FEATURES = [
    "sleep_duration_min",
    "sleep_efficiency",
    "wake_after_sleep_min",
    "bedtime_min",
    "home_time_min",
    "bt_unique_devices",
    "screen_night_min",
    "screen_total_min",
    "screen_unlocks",
]

# Raw per-day numeric features present for hrv rows (tier 2).
HRV_FEATURES = [
    "in_bed_duration",
    "sleep_duration",
    "sleep_latency",
    "waso",
    "bedtime",
    "HR",
    "lf_hf",
    "pnn50",
    "rmssd",
    "sdnn",
]

# Fields carried on every record that are NOT behavioral features -- label /
# identity / metadata columns.
ID_FIELDS = ["sample_id", "subject_id", "source", "tier", "window", "granularity"]
LABEL_FIELDS = ["phq4_score", "risk_category"]

RISK_ORDER = ["minimal", "mild", "moderate", "severe"]
RISK_TO_INT = {r: i for i, r in enumerate(RISK_ORDER)}

# `hrs` is near-perfectly collinear with risk_category in this batch (see
# project data-audit doc). It is excluded from the default feature set and
# only reintroduced in the explicit leakage-demonstration experiment.
LEAKAGE_SUSPECT_FIELDS = ["hrs"]
