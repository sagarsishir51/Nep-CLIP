"""
PHASE 4 — Analysis of the completed rater sheets.

Inputs : rater_sheets/rater_01.xlsx ... rater_10.xlsx  (filled by raters)
         assignment_key.csv                            (hidden key: dataset, anchor)
Outputs: printed report + resolved_scores.csv

Steps: load -> merge key -> attention-check QC -> reliability (alpha) on anchor
       -> rater outlier check -> resolve per-pair -> aggregate by dataset -> errors
"""
import glob
import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import kruskal
import krippendorff

N_RATERS = 10
K = 3                 # expected raters per distributed pair
ANCHOR_RATERS = 10    # expected raters per anchor pair

# ------------------------------------------ 0. verify assignment integrity
def verify_assignment(key_path="assignment_key.csv"):
    """Confirm anchor pairs -> all raters, distributed -> exactly K, even load."""
    k = pd.read_csv(key_path)
    rpp = k.groupby("pair_id")["rater"].nunique()
    is_anchor = k.groupby("pair_id")["is_anchor"].first()
    anchor, dist = rpp[is_anchor], rpp[~is_anchor]
    load = k.groupby("rater")["pair_id"].nunique()

    checks = {
        f"anchor pairs rated by all {ANCHOR_RATERS}": bool((anchor == ANCHOR_RATERS).all()),
        f"distributed pairs rated by exactly {K}": bool((dist == K).all()),
        "no pair rated twice by same rater":
            bool((k.groupby(["pair_id", "rater"]).size() == 1).all()),
        "rater load balanced (spread <= 1)": bool(load.max() - load.min() <= 1),
    }
    print("=== ASSIGNMENT INTEGRITY ===")
    print(f"  anchor pairs: {len(anchor)}  |  distributed pairs: {len(dist)}"
          f"  |  total: {len(rpp)}")
    print(f"  rater load: min={load.min()}, max={load.max()}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        print("  WARNING: assignment integrity check FAILED — "
              "do not distribute these sheets.")
    print()
    return all(checks.values())

verify_assignment()

# ---------------------------------------------------------- 1. load sheets
records = []
for f in sorted(glob.glob("rater_sheets/rater_*.xlsx")):
    r = int(f.split("_")[-1].split(".")[0])
    ws = openpyxl.load_workbook(f)["Ratings"]
    for row in range(2, ws.max_row + 1):
        pid = ws.cell(row, 1).value
        records.append({
            "rater": r, "pair_id": pid,
            "adequacy": ws.cell(row, 4).value,
            "fluency":  ws.cell(row, 5).value,
            "error_type": ws.cell(row, 6).value,
        })
R = pd.DataFrame(records)
print(f"Loaded {len(R)} ratings from {R.rater.nunique()} raters")

# --------------------------------------------- 2. attention-check QC
att = R[R.pair_id.isin(["ATT_G", "ATT_B"])]
fails = []
for r, g in att.groupby("rater"):
    good = g[g.pair_id == "ATT_G"]["adequacy"].values
    bad  = g[g.pair_id == "ATT_B"]["adequacy"].values
    ok = (len(good) and good[0] >= 4) and (len(bad) and bad[0] <= 2)
    if not ok:
        fails.append(r)
print(f"\nAttention-check FAILURES: {fails if fails else 'none'}")
R = R[~R.pair_id.isin(["ATT_G", "ATT_B"])]              # drop ATT rows
if fails:
    R = R[~R.rater.isin(fails)]                         # drop failing raters
    print(f"Excluded raters {fails}; {R.rater.nunique()} raters remain")

# -------------------------------------------------- 3. merge hidden key
key = pd.read_csv("assignment_key.csv")[["rater", "pair_id", "is_anchor", "dataset"]]
R = R.merge(key, on=["rater", "pair_id"], how="left")

# ------------------------------------------ 4. reliability on anchor set
def alpha(df, col):
    """Krippendorff alpha: build raters x items matrix for anchor pairs."""
    lvl = "ordinal" if col in ("adequacy", "fluency") else "nominal"
    d = df.copy()
    if col == "error_type":                       # encode categories to ints
        cats = {c: i for i, c in enumerate(pd.unique(d[col].dropna()))}
        d[col] = d[col].map(cats)
    piv = d.pivot_table(index="rater", columns="pair_id", values=col, aggfunc="first")
    data = piv.values.astype(float)
    return krippendorff.alpha(reliability_data=data, level_of_measurement=lvl)

anchor = R[R.is_anchor == True]
print("\n=== RELIABILITY (anchor set, all raters) ===")
for col in ("adequacy", "fluency", "error_type"):
    try:
        print(f"  Krippendorff alpha ({col}): {alpha(anchor, col):.3f}")
    except Exception as e:
        print(f"  alpha ({col}): n/a ({e})")

# within +/-1 agreement on anchor adequacy (pairwise avg)
piv = anchor.pivot_table(index="rater", columns="pair_id", values="adequacy")
pairs_within1 = []
raters = piv.index.tolist()
for i in range(len(raters)):
    for j in range(i+1, len(raters)):
        a, b = piv.iloc[i], piv.iloc[j]
        both = (~a.isna()) & (~b.isna())
        if both.sum():
            pairs_within1.append(((a-b).abs()[both] <= 1).mean())
print(f"  Adequacy within +/-1 (avg pairwise): {np.mean(pairs_within1):.1%}")

# ---------------------------------------- 5. rater outlier check
print("\n=== RATER OUTLIER CHECK (anchor adequacy vs group mean) ===")
grp_mean = piv.mean(axis=0)
for r in piv.index:
    corr = piv.loc[r].corr(grp_mean)
    flag = "  <-- low" if (corr is not None and corr < 0.3) else ""
    print(f"  rater {r}: corr with group = {corr:.2f}{flag}")

# ------------------------------------- 6. resolve per-pair scores
resolved = (R.groupby(["pair_id", "dataset"])
              .agg(adequacy=("adequacy", "mean"),
                   fluency=("fluency", "mean"),
                   n=("adequacy", "size"),
                   error_type=("error_type",
                               lambda s: s.value_counts().idxmax() if len(s) else "none"))
              .reset_index())
resolved.to_csv("resolved_scores.csv", index=False)

# ------------------------------------- 7. aggregate findings
print("\n=== MEAN SCORES BY DATASET ===")
by_ds = resolved.groupby("dataset").agg(
    n=("pair_id","size"), adequacy=("adequacy","mean"),
    fluency=("fluency","mean")).round(2)
print(by_ds.to_string())

# Kruskal-Wallis across datasets (adequacy)
groups = [g["adequacy"].values for _, g in resolved.groupby("dataset")]
H, p = kruskal(*groups)
print(f"\nKruskal-Wallis across datasets (adequacy): H={H:.2f}, p={p:.4f}")

print("\n=== ERROR-TYPE DISTRIBUTION ===")
ed = resolved["error_type"].value_counts()
ep = (ed/len(resolved)*100).round(1)
print(pd.DataFrame({"count": ed, "percent": ep}).to_string())

print("\nWrote resolved_scores.csv")