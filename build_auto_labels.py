from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(x) -> str:
    if pd.isna(x):
        x = ""
    x = str(x).lower()
    x = x.replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def count_pos_neg(
    text: str,
    positives: list[tuple[str, float]],
    negatives: list[tuple[str, float]],
) -> float:
    score = 0.0
    for pat, w in positives:
        if re.search(pat, text):
            score += w
    for pat, w in negatives:
        if re.search(pat, text):
            score -= w
    return score


# Updated positive and negative patterns for each task
VENT_POS = [
    (r"\bintubat(?:ed|ion)\b", 3.0),
    (r"\bendotracheal tube\b", 3.0),
    (r"\bett\b", 2.5),
    (r"\bmechanical ventilation\b", 3.0),
    (r"\bventilator\b", 3.0),
    (r"\btracheostom(?:y|ies)\b", 2.5),
    (r"\btrach\b", 2.0),
    (r"\bbipap\b", 2.0),
    (r"\bcpap\b", 1.8),
    (r"\brespiratory failure\b", 1.8),
    (r"\brespiratory distress\b", 1.5),
    (r"\bacute respiratory distress\b", 2.0),
    (r"\bhypoxi(?:a|emia)\b", 1.0),
    (r"\blow lung volumes\b", 0.3),
    (r"\bdyspnea\b", 0.5),
    (r"\bshortness of breath\b", 0.5),
    (r"\bsob\b", 0.5),
    (r"\bbilateral infiltrates\b", 1.0),
    (r"\bmultifocal pneumonia\b", 1.0),
    (r"\bdiffuse (?:airspace )?opacit(?:y|ies)\b", 1.0),
    (r"\bbibasilar airspace opacit(?:y|ies)\b", 0.8),
    (r"\bairspace disease\b", 0.8),
    (r"\bconsolidation\b", 0.8),
    (r"\bpleural effusions?\b", 0.5),
    (r"\bpulmonary edema\b", 1.0),
]
VENT_NEG = [
    (r"\bno (?:evidence of )?intubation\b", 2.0),
    (r"\bno (?:evidence of )?respiratory failure\b", 2.0),
    (r"\blungs are clear\b", 0.4),
    (r"\bno focal airspace disease\b", 0.4),
]

ICU_POS = [
    (r"\bicu\b", 3.0),
    (r"\bcritical care\b", 2.5),
    (r"\bmicu\b", 2.5),
    (r"\bsicu\b", 2.5),
    (r"\bccu\b", 2.5),
    (r"\brespiratory failure\b", 2.5),
    (r"\bshock\b", 2.5),
    (r"\bsepsis\b", 2.0),
    (r"\bhypoxi(?:a|emia)\b", 1.2),
    (r"\bdyspnea\b", 0.8),
    (r"\bshortness of breath\b", 0.8),
    (r"\bsob\b", 0.8),
    (r"\btachycardia\b", 0.8),
    (r"\bpulmonary edema\b", 1.8),
    (r"\binterstitial edema\b", 1.2),
    (r"\bvascular congestion\b", 1.2),
    (r"\bpulmonary vascular engorgement\b", 1.2),
    (r"\bcongestive heart failure\b", 1.8),
    (r"\bheart failure\b", 1.5),
    (r"\bcardiomegaly\b", 0.6),
    (r"\bpleural effusions?\b", 0.8),
    (r"\bbilateral pleural effusions?\b", 1.2),
    (r"\bmultifocal pneumonia\b", 1.5),
    (r"\bbilateral infiltrates\b", 1.5),
    (r"\bbibasilar airspace opacit(?:y|ies)\b", 1.0),
    (r"\blower lobe consolidation\b", 1.0),
    (r"\bconsolidation\b", 0.8),
    (r"\bairspace disease\b", 0.8),
    (r"\bintubat(?:ed|ion)\b", 2.5),
    (r"\bendotracheal tube\b", 2.5),
    (r"\bventilator\b", 2.5),
    (r"\bbipap\b", 1.2),
    (r"\bcpap\b", 1.0),
]
ICU_NEG = [
    (r"\bno (?:evidence of )?pulmonary edema\b", 1.5),
    (r"\bnegative for (?:pulmonary edema|vascular congestion)\b", 1.5),
    (r"\bno focal airspace disease\b", 0.6),
    (r"\blungs are clear\b", 0.6),
]

MORT_POS = [
    (r"\bexpired\b", 5.0),
    (r"\bdeceased\b", 5.0),
    (r"\bpostmortem\b", 5.0),
    (r"\bhospice\b", 4.0),
    (r"\bcomfort care\b", 4.0),
    (r"\bcode blue\b", 4.0),
    (r"\barrest\b", 3.5),
    (r"\bshock\b", 3.0),
    (r"\bsepsis\b", 2.0),
    (r"\brespiratory failure\b", 2.5),
    (r"\bintubat(?:ed|ion)\b", 2.5),
    (r"\bendotracheal tube\b", 2.5),
    (r"\bventilator\b", 2.5),
    (r"\bhypoxi(?:a|emia)\b", 1.0),
    (r"\bdyspnea\b", 0.5),
    (r"\bshortness of breath\b", 0.5),
    (r"\bsob\b", 0.5),
    (r"\bdiffuse pulmonary edema\b", 2.5),
    (r"\bpulmonary edema\b", 1.5),
    (r"\binterstitial edema\b", 1.2),
    (r"\bvascular congestion\b", 1.0),
    (r"\bpulmonary vascular engorgement\b", 1.2),
    (r"\blarge bilateral pleural effusions?\b", 2.0),
    (r"\bbilateral pleural effusions?\b", 1.2),
    (r"\bpleural effusions?\b", 0.8),
    (r"\bmultifocal pneumonia\b", 1.5),
    (r"\bbilateral infiltrates\b", 1.2),
    (r"\bmultifocal (?:airspace )?opacit(?:y|ies)\b", 1.2),
    (r"\bbilateral (?:airspace )?opacit(?:y|ies)\b", 1.2),
    (r"\bdiffuse (?:airspace )?opacit(?:y|ies)\b", 1.2),
    (r"\bconsolidation\b", 0.8),
    (r"\bairspace disease\b", 0.8),
    (r"\bcardiomegaly\b", 0.8),
    (r"\bheart is enlarged\b", 0.8),
    (r"\bpericardial effusion\b", 1.5),
]
MORT_NEG = [
    (r"\bno (?:evidence of )?pulmonary edema\b", 1.2),
    (r"\blungs are clear\b", 0.8),
    (r"\bno focal airspace disease\b", 0.8),
    (r"\bno pleural effusion\b", 0.6),
]


def build_labels_row(notes, report) -> tuple[int, int, int, float, float, float]:
    notes = "" if pd.isna(notes) else str(notes)
    report = "" if pd.isna(report) else str(report)

    text = normalize_text(notes + " " + report)

    vent_score = count_pos_neg(text, VENT_POS, VENT_NEG)
    icu_score = count_pos_neg(text, ICU_POS, ICU_NEG)
    mort_score = count_pos_neg(text, MORT_POS, MORT_NEG)

    if vent_score >= 2.5:
        icu_score += 1.0
        mort_score += 0.8

    if icu_score >= 1.5:
        mort_score += 0.5

    if re.search(r"\b(respiratory failure|shock|sepsis)\b", text) and re.search(
        r"\b(intubat(?:ed|ion)|ventilator|endotracheal tube)\b", text
    ):
        mort_score += 1.5
        icu_score += 1.0

    if re.search(r"\b(dyspnea|shortness of breath|sob)\b", text) and re.search(
        r"\b(vascular congestion|pulmonary edema|pleural effusion|consolidation|airspace opacity|airspace disease)\b",
        text,
    ):
        icu_score += 0.8
        mort_score += 0.6

    ventilation_risk = int(vent_score >= 1.0)
    icu_risk = int(icu_score >= 1.0)
    mortality_risk = int(mort_score >= 1.2)

    if re.search(r"\b(intubated|endotracheal tube|ett|ventilator|mechanical ventilation|bipap|cpap)\b", text):
        ventilation_risk = 1
        icu_risk = 1

    return mortality_risk, icu_risk, ventilation_risk, mort_score, icu_score, vent_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-studies-csv", default="artifacts/studies.csv")
    parser.add_argument("--output-csv", default="artifacts/labels.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_studies_csv)

    if "study_id" not in df.columns:
        raise RuntimeError("studies.csv 必须有 study_id 列")

    if "subject_id" not in df.columns:
        df["subject_id"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""
    if "report" not in df.columns:
        df["report"] = ""

    morts, icus, vents = [], [], []
    mort_scores, icu_scores, vent_scores = [], [], []

    for _, row in df.iterrows():
        m, i, v, ms, is_, vs = build_labels_row(row.get("notes", ""), row.get("report", ""))
        morts.append(m)
        icus.append(i)
        vents.append(v)
        mort_scores.append(ms)
        icu_scores.append(is_)
        vent_scores.append(vs)

    out = df[["study_id", "subject_id", "notes", "report"]].copy()
    out["mortality_risk"] = morts
    out["icu_risk"] = icus
    out["ventilation_risk"] = vents
    out["_mort_score"] = mort_scores
    out["_icu_score"] = icu_scores
    out["_vent_score"] = vent_scores

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Saved to {args.output_csv}")
    print("shape =", out.shape)
    print("positives:")
    print("  mortality_risk   =", int(out["mortality_risk"].sum()))
    print("  icu_risk         =", int(out["icu_risk"].sum()))
    print("  ventilation_risk =", int(out["ventilation_risk"].sum()))
    print("rates:")
    print("  mortality_risk   =", float(out["mortality_risk"].mean()))
    print("  icu_risk         =", float(out["icu_risk"].mean()))
    print("  ventilation_risk =", float(out["ventilation_risk"].mean()))


if __name__ == "__main__":
    main()