"""
USDA FDC Auto-Matcher
=====================
Reads ingredients_database.csv, queries USDA FoodData Central for each ingredient,
finds the closest match by calorie/macro profile, and fills in the usda_fdc_id column.

Output:
  - ingredients_database.csv        : updated with usda_fdc_id filled in
  - usda_review_report.csv          : flagged rows (>10% calorie diff) for human review
  - usda_auto_approved.csv          : rows auto-approved (<=10% calorie diff)

Usage:
  1. Get a FREE API key at https://fdc.nal.usda.gov/api-guide.html  (takes ~1 min)
     ⚠️  Do NOT use DEMO_KEY — it allows only 10 req/hour (not enough for 550 ingredients)
     A real key allows 1000 req/hour — the script will finish in ~10 minutes.

  2. export USDA_API_KEY=your_key_here
     python match_usda_ids.py

Reviewers:
  - Karunya  : confirm ingredient context / intended food type
  - Backend developer : confirm USDA entry makes sense from the API side
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY         = os.environ.get("USDA_API_KEY", "")
SEARCH_URL      = "https://api.nal.usda.gov/fdc/v1/foods/search"
DATA_TYPES      = "SR Legacy,Foundation"
FLAG_THRESHOLD  = 0.10   # flag if calorie diff > 10%
RATE_LIMIT_S    = 3.6    # seconds between calls → ~1000 req/hour (real API key limit)

BASE_DIR        = Path(__file__).parent
INPUT_CSV       = BASE_DIR / "ingredients_database.csv"
OUTPUT_CSV      = BASE_DIR / "ingredients_database.csv"
REPORT_FLAGGED  = BASE_DIR / "usda_review_report.csv"
REPORT_APPROVED = BASE_DIR / "usda_auto_approved.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def usda_search(query: str) -> list[dict]:
    """Search USDA by ingredient name. Returns list of food entries."""
    params = urllib.parse.urlencode({
        "query":    query,
        "api_key":  API_KEY,
        "pageSize": 10,
        "dataType": DATA_TYPES,
    })
    url = f"{SEARCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("foods", [])
    except Exception as e:
        print(f"  [WARN] Search failed for '{query}': {e}")
        return []


def get_nutrients(food: dict) -> dict:
    """Extract key nutrients (per 100g) from a search result food entry."""
    raw = {n["nutrientName"]: n.get("value", 0) for n in food.get("foodNutrients", []) if "nutrientName" in n}

    # Energy: prefer kcal entry (value < 900 is almost certainly kcal, not kJ)
    energy_candidates = [v for k, v in raw.items() if "Energy" in k and v < 900]
    calories = energy_candidates[0] if energy_candidates else 0.0

    return {
        "calories":  calories,
        "fat":       raw.get("Total lipid (fat)", 0.0),
        "carbs":     raw.get("Carbohydrate, by difference", 0.0),
        "protein":   raw.get("Protein", 0.0),
        "fiber":     raw.get("Fiber, total dietary", 0.0),
    }


def best_match(foods: list[dict], target: dict) -> tuple[dict | None, dict | None]:
    """
    Score each USDA result against the target nutrients.
    Uses weighted Euclidean distance (calories weighted highest).
    Returns (best_food, best_nutrients).
    """
    weights = {"calories": 2.0, "fat": 1.0, "carbs": 1.0, "protein": 1.5, "fiber": 0.5}

    best_food  = None
    best_nut   = None
    best_score = float("inf")

    for food in foods:
        nut = get_nutrients(food)
        if nut["calories"] == 0:
            continue

        score = sum(
            weights[k] * ((nut[k] - target[k]) ** 2)
            for k in weights
        )

        if score < best_score:
            best_score = score
            best_food  = food
            best_nut   = nut

    return best_food, best_nut


def calorie_diff_pct(csv_cal: float, usda_cal: float) -> float:
    """Percentage difference in calories between CSV and USDA match."""
    if csv_cal == 0:
        return 0.0
    return abs(csv_cal - usda_cal) / csv_cal


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("❌ USDA_API_KEY is not set.")
        print("   Get a free key at https://fdc.nal.usda.gov/api-guide.html")
        print("   Then run:  export USDA_API_KEY=your_key_here")
        return

    print(f"Reading {INPUT_CSV} ...")
    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "usda_fdc_id" not in fieldnames:
        fieldnames.append("usda_fdc_id")

    flagged  = []
    approved = []
    updated  = []

    total = len(rows)
    for i, row in enumerate(rows):
        ingredient = row["ingredient"]
        existing_id = row.get("usda_fdc_id", "").strip()

        # Skip if already pinned
        if existing_id:
            print(f"[{i+1}/{total}] SKIP (already pinned) : {ingredient} → {existing_id}")
            updated.append(row)
            continue

        # Target nutrients per 100g (CSV stores per gram → ×100)
        try:
            target = {
                "calories": float(row["calories_per_g"]) * 100,
                "fat":      float(row["fat_per_g"])      * 100,
                "carbs":    float(row["carbs_per_g"])    * 100,
                "protein":  float(row["protein_per_g"])  * 100,
                "fiber":    float(row["fiber_per_g"])    * 100,
            }
        except ValueError:
            print(f"[{i+1}/{total}] SKIP (bad data)       : {ingredient}")
            updated.append(row)
            continue

        # Query USDA
        foods = usda_search(ingredient)
        time.sleep(RATE_LIMIT_S)

        if not foods:
            print(f"[{i+1}/{total}] NO RESULT            : {ingredient}")
            row["usda_fdc_id"] = ""
            updated.append(row)
            flagged.append({**row, "reason": "no USDA result found", "usda_calories": ""})
            continue

        match, match_nut = best_match(foods, target)

        if not match:
            print(f"[{i+1}/{total}] NO MATCH             : {ingredient}")
            row["usda_fdc_id"] = ""
            updated.append(row)
            flagged.append({**row, "reason": "no usable USDA nutrients", "usda_calories": ""})
            continue

        fdc_id     = match["fdcId"]
        usda_cal   = match_nut["calories"]
        diff_pct   = calorie_diff_pct(target["calories"], usda_cal)
        diff_label = f"{diff_pct * 100:.1f}%"

        row["usda_fdc_id"] = str(fdc_id)
        updated.append(row)

        report_row = {
            **row,
            "usda_description": match["description"],
            "csv_calories_per100g":  round(target["calories"], 1),
            "usda_calories_per100g": round(usda_cal, 1),
            "calorie_diff_pct":      diff_label,
        }

        if diff_pct > FLAG_THRESHOLD:
            status = "FLAG  "
            flagged.append(report_row)
        else:
            status = "OK    "
            approved.append(report_row)

        print(f"[{i+1}/{total}] {status} {ingredient:35s} → fdcId={fdc_id} | csv={target['calories']:.0f} usda={usda_cal:.0f} diff={diff_label} | {match['description']}")

    # ── Write updated CSV ──────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    print(f"\n✅ Updated CSV saved → {OUTPUT_CSV}")

    # ── Write flagged report ───────────────────────────────────────────────────
    report_fields = fieldnames + ["usda_description", "csv_calories_per100g", "usda_calories_per100g", "calorie_diff_pct", "reason"]
    report_fields = list(dict.fromkeys(report_fields))  # deduplicate

    with open(REPORT_FLAGGED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flagged)
    print(f"🚩 Flagged rows (need review) → {REPORT_FLAGGED}  ({len(flagged)} items)")

    with open(REPORT_APPROVED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(approved)
    print(f"✅ Auto-approved rows         → {REPORT_APPROVED}  ({len(approved)} items)")

    print(f"\nSummary: {len(approved)} auto-approved | {len(flagged)} need review | {total} total")


if __name__ == "__main__":
    main()
