"""
Build TimeSRL-style prompts from the raw daily behavioral dicts.

Per the TimeSRL paper (arXiv:2605.21295), Stage-1 input is a structured
prompt with daily feature vectors ("Sleep Duration (hours): {D1: 6.5, D2:
7.2, ...}"), descriptive feature names, and units -- not pre-aggregated
stats. So this reads the RAW per-day dicts from the original JSON (not the
engineered mean/std/pctile table from src/build_dataset.py), to match the
paper's actual input format as closely as the data allows.

Scope: tier-1 GLOBEM only, same "clean" slice as src/build_clean_dataset.py
-- see that file for why tier 3 / hrv / hrs are excluded.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "anchored_globem_hrv.json"

# raw_field -> (descriptive label, unit, scale factor applied to the raw value)
FEATURE_SPECS = {
    "sleep_duration_min": ("Sleep Duration", "hours", 1 / 60),
    "sleep_efficiency": ("Sleep Efficiency", "%", 1),
    "wake_after_sleep_min": ("Wake After Sleep Onset", "minutes", 1),
    "bedtime_min": ("Bedtime", "minutes after midnight", 1),
    "home_time_min": ("Time Spent at Home", "hours", 1 / 60),
    "bt_unique_devices": ("Unique Nearby Bluetooth Devices", "count", 1),
    "screen_night_min": ("Nighttime Screen Time", "minutes", 1),
    "screen_total_min": ("Total Daily Screen Time", "hours", 1 / 60),
    "screen_unlocks": ("Phone Unlocks", "count", 1),
}

RISK_LABELS = ["minimal", "mild", "moderate", "severe"]

INSTRUCTION = (
    "You are reviewing one week of passive smartphone/wearable sensing data "
    "for a single study participant, day D1 (oldest) through D7 (most "
    "recent). A blank value means no reading was recorded that day.\n\n"
    "{feature_block}\n"
    "Task: first, in 2-4 sentences, write a plain-language clinical summary "
    "of this person's sleep, activity, phone-use and mobility patterns over "
    "the week. Then, on its own final line, give your depression-risk "
    "assessment in exactly this format:\n"
    "Risk score: <0, 1, 2, or 3>\n"
    "where 0=minimal, 1=mild, 2=moderate, 3=severe."
)


def _fmt_day(v, scale):
    if v is None:
        return "null"
    v = v * scale
    return f"{v:.1f}" if scale != 1 else (f"{v:.0f}" if isinstance(v, float) and v == int(v) else f"{v:.1f}")


def build_feature_block(record):
    lines = []
    for raw_field, (label, unit, scale) in FEATURE_SPECS.items():
        day_dict = record.get(raw_field) or {}
        days = {d: _fmt_day(v, scale) for d, v in day_dict.items()}
        inner = ", ".join(f"{d}: {days.get(d, 'null')}" for d in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"])
        lines.append(f"{label} ({unit}): {{{inner}}}")
    return "\n".join(lines)


def load_clean_tier1_records():
    with open(RAW_PATH) as f:
        raw = json.load(f)
    return [r for r in raw if r["tier"] == 1]


def build_examples():
    """Returns a list of dicts: {sample_id, subject_id, prompt, risk_int, risk_category}."""
    records = load_clean_tier1_records()
    examples = []
    for rec in records:
        feature_block = build_feature_block(rec)
        prompt = INSTRUCTION.format(feature_block=feature_block)
        examples.append({
            "sample_id": rec["sample_id"],
            "subject_id": rec["subject_id"],
            "prompt": prompt,
            "risk_category": rec["risk_category"],
            "risk_int": RISK_LABELS.index(rec["risk_category"]),
        })
    return examples


if __name__ == "__main__":
    examples = build_examples()
    print(f"Built {len(examples)} tier-1 prompts.")
    ex = examples[0]
    print(f"\n--- example prompt (sample_id={ex['sample_id']}, true risk={ex['risk_category']}) ---\n")
    print(ex["prompt"])
    lens = [len(e["prompt"]) for e in examples]
    print(f"\nprompt char length: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")
