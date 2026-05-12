"""nlaB3_compare.py -- post-hoc analysis of NLA descriptions produced by
nlaB2_run_nla.py.

Reads results/nlaB_descriptions.json and produces:
  - results/nlaB_table.tsv: per-case table of NLA descriptions across the
    7 extracted token positions (NF-content + NL-content/pregen/A/B/C/D)
  - results/nlaB_summary.json: structured per-kind statistics:
      - mean description length
      - keyword-tally agreement with hypotheses:
          * "content" should mention medical/clinical/symptom/patient...
          * "pregen"/"letter_*" should mention letter/answer/format/
            instruction/multiple-choice/scaffold/option...

The analysis is intentionally simple — keyword tallies plus a small
exemplar table. The point is to show whether NLA's natural-language
descriptions agree with our top-token analysis of Phase 5, NOT to
quantify NLA quality in detail.

Run from project root (after nlaB2 completes locally or via pull):
    python3 paper/scripts/nlaB3_compare.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIONS = ROOT / "results" / "nlaB_descriptions.json"
OUT_TABLE = ROOT / "results" / "nlaB_table.tsv"
OUT_SUMMARY = ROOT / "results" / "nlaB_summary.json"

# Keyword pools for hypothesis testing
MEDICAL_KW = {
    "medical", "clinical", "doctor", "physician", "patient", "symptom",
    "symptoms", "disease", "diagnosis", "treatment", "hospital", "emergency",
    "health", "healthcare", "illness", "pain", "fever", "blood", "heart",
    "neurological", "infection", "consultation", "appointment", "follow-up",
    "history-taking", "anamnesis", "presenting", "complaint",
}
FORMAT_KW = {
    "letter", "letters", "answer", "answers", "format", "instruction",
    "instructions", "multiple-choice", "option", "options", "label",
    "labels", "scaffold", "scaffolding", "response", "constrained",
    "structured", "list", "enumeration", "enumerated", "choice", "choices",
    "selection", "multiple choice", "exam", "test", "question-answering",
}


def has_any_kw(text: str, kws: set[str]) -> list[str]:
    """Return the list of keywords present in text (case-insensitive,
    word-boundary aware). Multi-word KWs use substring match."""
    t = text.lower()
    found = []
    for kw in kws:
        if " " in kw or "-" in kw:
            if kw in t:
                found.append(kw)
        else:
            # word-boundary match for single-word KWs
            if re.search(rf"\b{re.escape(kw)}\b", t):
                found.append(kw)
    return found


def main():
    if not DESCRIPTIONS.exists():
        raise SystemExit(f"missing {DESCRIPTIONS} -- run nlaB2_run_nla.py first")
    data = json.loads(DESCRIPTIONS.read_text())
    results = data["results"]
    print(f"[B3] loaded {len(results)} NLA description records")

    # Bucket by (case_id, format, kind)
    by_case = defaultdict(dict)
    for r in results:
        # We took the first sample (greedy / 1-sample) for the headline tally
        text = r["samples"][0] if r["samples"] else ""
        by_case[r["case_id"]][(r["format"], r["kind"])] = text

    # Build the wide TSV table
    cols = ["case_id",
            "NF_content", "NL_content", "NL_pregen",
            "NL_letter_A", "NL_letter_B", "NL_letter_C", "NL_letter_D"]
    rows = []
    for cid, kinds in sorted(by_case.items()):
        row = {"case_id": cid}
        row["NF_content"]  = kinds.get(("NF", "content"), "")
        row["NL_content"]  = kinds.get(("NL", "content"), "")
        row["NL_pregen"]   = kinds.get(("NL", "pregen"), "")
        row["NL_letter_A"] = kinds.get(("NL", "letter_A"), "")
        row["NL_letter_B"] = kinds.get(("NL", "letter_B"), "")
        row["NL_letter_C"] = kinds.get(("NL", "letter_C"), "")
        row["NL_letter_D"] = kinds.get(("NL", "letter_D"), "")
        rows.append(row)

    with OUT_TABLE.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(
                row[c].replace("\t", " ").replace("\n", " | ") for c in cols
            ) + "\n")
    print(f"[B3] wrote per-case table -> {OUT_TABLE}")

    # ─── Keyword tally per kind ──────────────────────────────────────────
    kinds_to_check = [
        ("NF", "content"),
        ("NL", "content"),
        ("NL", "pregen"),
        ("NL", "letter_A"),
        ("NL", "letter_B"),
        ("NL", "letter_C"),
        ("NL", "letter_D"),
    ]
    summary = {}
    for fmt, kind in kinds_to_check:
        texts = [by_case[cid].get((fmt, kind), "") for cid in by_case]
        n = len(texts)
        n_medical = sum(1 for t in texts if has_any_kw(t, MEDICAL_KW))
        n_format = sum(1 for t in texts if has_any_kw(t, FORMAT_KW))
        med_kws = Counter()
        fmt_kws = Counter()
        for t in texts:
            for k in has_any_kw(t, MEDICAL_KW): med_kws[k] += 1
            for k in has_any_kw(t, FORMAT_KW):  fmt_kws[k] += 1
        avg_len = sum(len(t.split()) for t in texts) / max(n, 1)
        summary[f"{fmt}_{kind}"] = {
            "n_cases": n,
            "n_with_medical_kw": n_medical,
            "n_with_format_kw": n_format,
            "pct_medical": round(100 * n_medical / max(n, 1), 1),
            "pct_format":  round(100 * n_format / max(n, 1), 1),
            "avg_word_count": round(avg_len, 1),
            "top_medical_kws": med_kws.most_common(8),
            "top_format_kws":  fmt_kws.most_common(8),
        }

    # NL_content vs NF_content sanity check: under causal attention, the
    # token-level L32 activations should be identical, so the NLA outputs
    # at temperature=0 (greedy) should ALSO be identical. Tally the
    # exact-match rate.
    nf_content = {cid: by_case[cid].get(("NF", "content"), "") for cid in by_case}
    nl_content = {cid: by_case[cid].get(("NL", "content"), "") for cid in by_case}
    n_identical = sum(1 for cid in by_case
                      if nf_content[cid] == nl_content[cid] and nf_content[cid] != "")
    summary["sanity_NF_eq_NL_content"] = {
        "n_cases_total": len(by_case),
        "n_identical_descriptions": n_identical,
        "pct_identical": round(100 * n_identical / max(len(by_case), 1), 1),
        "comment": ("Under causal attention NL/NF content-token activations "
                    "must be byte-identical; greedy NLA decode should "
                    "therefore produce identical strings. Lower pct = some "
                    "non-determinism in the SGLang decode path."),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"[B3] wrote summary stats -> {OUT_SUMMARY}")

    # ─── Print headline numbers ──────────────────────────────────────────
    print("\n[B3] Headline tally (pct of cases mentioning each keyword pool):")
    print(f"  {'kind':<18}  {'med %':>6}  {'fmt %':>6}  {'avg_len':>8}")
    for fmt, kind in kinds_to_check:
        key = f"{fmt}_{kind}"
        s = summary[key]
        print(f"  {key:<18}  {s['pct_medical']:>6.1f}  {s['pct_format']:>6.1f}  {s['avg_word_count']:>8.1f}")
    s = summary["sanity_NF_eq_NL_content"]
    print(f"\n  Sanity (NL_content == NF_content under causal LM + greedy NLA):"
          f" {s['n_identical_descriptions']}/{s['n_cases_total']} "
          f"({s['pct_identical']}%)")


if __name__ == "__main__":
    main()
