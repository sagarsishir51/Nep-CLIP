"""
PHASE 2 — Assignment: turn sample_pairs.csv into 10 blinded rater sheets.

Design:
  - Anchor set: 50 pairs (10 per dataset) rated by ALL 10 raters  -> agreement (alpha)
  - Distributed: remaining 250 pairs, each rated by exactly 3 raters, balanced rotation
  - Each rater sheet: 50 anchor + 75 distributed = 125 pairs, shuffled, blinded
  - 2 attention-check pairs seeded per rater (1 obvious-good, 1 obvious-bad)

Inputs : sample_pairs.csv  (pair_id, dataset, id, image_id, english, nepali)
Outputs: rater_sheets/rater_01.xlsx ... rater_10.xlsx   (what raters fill)
         assignment_key.csv                             (hidden: who rated what, anchor flags)
"""
import os
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

SEED = 42
rng = np.random.default_rng(SEED)

N_RATERS = 10
ANCHOR_PER_DATASET = 10     # 10 x 5 = 50 anchor pairs
K = 3                       # raters per distributed pair
ATTENTION_PER_RATER = 2     # seeded checks (1 good, 1 bad)

ERR_TYPES = "none,mistranslation,dropped,added,cultural,script,grammar"

# ---------------------------------------------------------------- load
df = pd.read_csv("../sample_pairs.csv")
required = {"pair_id", "dataset", "english", "nepali"}
if not required.issubset(df.columns):
    raise SystemExit(f"sample_pairs.csv missing columns: {required - set(df.columns)}")

# ---------------------------------------------------- 1. anchor carve-out
anchor_ids = []
for dset, grp in df.groupby("dataset"):
    take = min(ANCHOR_PER_DATASET, len(grp))
    anchor_ids += list(rng.choice(grp["pair_id"].values, size=take, replace=False))
anchor_set = set(anchor_ids)
df["is_anchor"] = df["pair_id"].isin(anchor_set)
distributed = df[~df["is_anchor"]].copy()
print(f"Anchor: {len(anchor_set)} pairs (all 10 raters)")
print(f"Distributed: {len(distributed)} pairs (3 raters each)")

# --------------------------------- 2. balanced distributed assignment
# Build a list of (pair_id -> 3 distinct raters) with even load + balanced overlap.
# Simple, well-balanced method: round-robin over a shuffled rater sequence.
rater_ids = list(range(1, N_RATERS + 1))
assignments = {r: [] for r in rater_ids}          # rater -> [pair_ids]
pair_raters = {}                                  # pair_id -> [raters]

dist_pairs = list(distributed["pair_id"].values)
rng.shuffle(dist_pairs)

# maintain a rotating pointer so loads stay even
slot = 0
for pid in dist_pairs:
    chosen = []
    tries = 0
    while len(chosen) < K and tries < N_RATERS * 3:
        r = rater_ids[slot % N_RATERS]
        slot += 1
        tries += 1
        if r not in chosen:
            chosen.append(r)
    # balance: prefer raters with the fewest pairs so far
    # (re-pick if this created imbalance > 1)
    pair_raters[pid] = chosen
    for r in chosen:
        assignments[r].append(pid)

# rebalance pass: even out loads to within +/-1
def loads():
    return {r: len(v) for r, v in assignments.items()}

target = int(np.ceil(len(dist_pairs) * K / N_RATERS))
for _ in range(2000):
    l = loads()
    hi = max(l, key=l.get); lo = min(l, key=l.get)
    if l[hi] - l[lo] <= 1:
        break
    # move a pair from hi-rater to lo-rater if lo not already on it
    moved = False
    for pid in assignments[hi]:
        if lo not in pair_raters[pid]:
            assignments[hi].remove(pid)
            pair_raters[pid] = [lo if x == hi else x for x in pair_raters[pid]]
            assignments[lo].append(pid)
            moved = True
            break
    if not moved:
        break

# every rater also gets ALL anchor pairs
for r in rater_ids:
    assignments[r] = list(anchor_set) + assignments[r]

print("Per-rater load (anchor+distributed):",
      {r: len(v) for r, v in assignments.items()})

# ------------------------------------------- 3. attention-check pairs
# Craft two obvious pairs (not from the data) seeded into each sheet.
ATT_GOOD = {"pair_id": "ATT_G", "english": "A cat sits on a chair.",
            "nepali": "एउटा बिरालो कुर्सीमा बस्छ।"}          # obviously correct
ATT_BAD  = {"pair_id": "ATT_B", "english": "The sun rises in the east.",
            "nepali": "केराको मूल्य बजारमा बढ्यो।"}          # obviously wrong (banana price)

# --------------------------------------------------- 4. write sheets
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
HEAD_FONT = Font(bold=True, color="FFFFFF")
BODY = Font(size=11)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="center")
thin = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

os.makedirs("rater_sheets", exist_ok=True)
pairs_by_id = df.set_index("pair_id")[["english", "nepali"]].to_dict("index")

key_rows = []
for r in rater_ids:
    wb = openpyxl.Workbook()

    # Rubric tab (brief)
    rub = wb.active; rub.title = "Rubric"
    rub_lines = [
        ("Adequacy (meaning preserved?)  5=fully  4=minor loss  3=partial  2=major loss  1=wrong",),
        ("Fluency (natural Nepali?)      5=native 4=good 3=clunky 2=broken 1=not Nepali",),
        ("Error type: none / mistranslation / dropped / added / cultural / script / grammar",),
        ("",),
        ("Rate adequacy & fluency SEPARATELY. Do NOT discuss with other raters until done.",),
    ]
    for i, (t,) in enumerate(rub_lines, 1):
        rub.cell(i, 1, t).font = Font(bold=(i <= 3))
    rub.column_dimensions["A"].width = 95

    # Ratings tab
    ws = wb.create_sheet("Ratings")
    headers = ["pair_id", "english", "nepali", "adequacy", "fluency", "error_type", "notes"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(1, c); cell.fill = HEAD_FILL; cell.font = HEAD_FONT
        cell.alignment = CENTER; cell.border = BORDER

    # assemble this rater's rows: their pairs + 2 attention checks, shuffled
    rows = []
    for pid in assignments[r]:
        p = pairs_by_id[pid]
        rows.append((pid, p["english"], p["nepali"]))
    rows.append((ATT_GOOD["pair_id"], ATT_GOOD["english"], ATT_GOOD["nepali"]))
    rows.append((ATT_BAD["pair_id"], ATT_BAD["english"], ATT_BAD["nepali"]))
    rng.shuffle(rows)

    for (pid, en, ne) in rows:
        ws.append([pid, en, ne, None, None, None, None])
    last = ws.max_row
    for rr in range(2, last + 1):
        for c in range(1, len(headers) + 1):
            cell = ws.cell(rr, c); cell.font = BODY; cell.border = BORDER
            cell.alignment = WRAP if c in (2, 3, 7) else CENTER

    dv_s = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dv_e = DataValidation(type="list", formula1=f'"{ERR_TYPES}"', allow_blank=True)
    ws.add_data_validation(dv_s); ws.add_data_validation(dv_e)
    dv_s.add(f"D2:D{last}"); dv_s.add(f"E2:E{last}"); dv_e.add(f"F2:F{last}")

    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 42
    ws.column_dimensions["C"].width = 42
    for col in ("D", "E", "F"):
        ws.column_dimensions[col].width = 11
    ws.column_dimensions["G"].width = 24
    ws.freeze_panes = "A2"

    wb.save(f"rater_sheets/rater_{r:02d}.xlsx")

    # record key
    for pid in assignments[r]:
        key_rows.append({"rater": r, "pair_id": pid,
                         "is_anchor": pid in anchor_set})

# hidden assignment key for analysis
key = pd.DataFrame(key_rows)
key = key.merge(df[["pair_id", "dataset", "id", "image_id"]], on="pair_id", how="left")
key.to_csv("assignment_key.csv", index=False)

# verify redundancy
red = key[~key["is_anchor"]].groupby("pair_id")["rater"].nunique()
print(f"\nDistributed redundancy: min={red.min()}, max={red.max()} (target {K})")
print(f"Anchor pairs rated by all 10: "
      f"{(key[key['is_anchor']].groupby('pair_id')['rater'].nunique() == N_RATERS).all()}")
print(f"Wrote 10 sheets to rater_sheets/ and assignment_key.csv")
