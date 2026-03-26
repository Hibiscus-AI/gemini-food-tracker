"""
Food Calorie Tracker API - Powered by Gemini + USDA + IFCT

Pipeline:
1. Gemini identifies food + ingredients from photo (what it's good at)
2. Nutrition waterfall: IFCT (Indian foods) → USDA → local DB → Gemini fallback

Serving sizes: bowl (~250g), plate (~350g), piece, cup (~240g)
"""

import os
import csv
import json
import re
import logging
import asyncio
import time
from io import BytesIO
from typing import Optional

import httpx
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from google import genai

# ============== LOGGING ==============

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("food-tracker")

# ============== CONFIGURATION ==============

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
MODEL_NAME = "gemini-2.0-flash"
MAX_IMAGE_SIZE_MB = 10
GEMINI_TIMEOUT_SECONDS = 15
CONFIDENCE_WARN_THRESHOLD = 0.6
MIN_INGREDIENT_GRAMS = 1
MAX_INGREDIENT_GRAMS = 2000

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f"Gemini configured ({MODEL_NAME})")
else:
    gemini_client = None
    logger.warning("GEMINI_API_KEY not set")

logger.info(f"USDA API key: {'custom' if USDA_API_KEY != 'DEMO_KEY' else 'DEMO_KEY (rate limited)'}")

# ============== SERVING SIZE PRESETS ==============

SERVING_SIZES = {
    "bowl": 250,
    "plate": 350,
    "cup": 240,
    "piece": None,  # varies by food, Gemini estimates
    "small_bowl": 150,
    "large_bowl": 400,
    "tablespoon": 15,
    "slice": 30,
    "glass": 250,
}

# ============== LOCAL INGREDIENT DATABASE ==============

DATASET_DIR = Path(__file__).parent.parent.parent / "goldendataset" / "metadata"

def load_local_ingredients_db():
    """Load local ingredients_database.csv as fallback."""
    db = {}
    csv_path = DATASET_DIR / "ingredients_database.csv"
    if not csv_path.exists():
        return db
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["ingredient"].strip().lower()
            db[name] = {
                "calories_per_g": float(row["calories_per_g"]),
                "fat_per_g": float(row["fat_per_g"]),
                "carbs_per_g": float(row["carbs_per_g"]),
                "protein_per_g": float(row["protein_per_g"]),
                "fiber_per_g": float(row["fiber_per_g"]),
                "source": "local_db",
            }
    logger.info(f"Local ingredient DB: {len(db)} ingredients")
    return db

LOCAL_INGREDIENTS_DB = load_local_ingredients_db()

# ============== IFCT (Indian Food) DATABASE ==============

def load_ifct_db():
    """Load IFCT Indian food nutrition database."""
    db = {}
    csv_path = DATASET_DIR / "ifct_ingredients_database.csv"
    if not csv_path.exists():
        return db
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["ingredient"].strip().lower()
            db[name] = {
                "calories_per_g": float(row["calories_per_g"]),
                "fat_per_g": float(row["fat_per_g"]),
                "carbs_per_g": float(row["carbs_per_g"]),
                "protein_per_g": float(row["protein_per_g"]),
                "fiber_per_g": float(row["fiber_per_g"]),
                "source": "ifct",
            }
    logger.info(f"IFCT Indian food DB: {len(db)} ingredients")
    return db

IFCT_DB = load_ifct_db()

# ============== MODELS ==============

class NutritionInfo(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: Optional[float] = 0

class IngredientNutrition(BaseModel):
    name: str
    estimated_grams: float
    nutrition: NutritionInfo
    source: str = "ai"  # "ifct", "usda", "local_db", or "ai"

class IngredientChip(BaseModel):
    """Simplified format for UI ingredient chips: 'Eggs - 24cal'"""
    name: str
    cal: int
    grams: float

class DishAnalysis(BaseModel):
    dish_name: str
    dish_type: str = "solid"  # "solid", "beverage", "packaged"
    ingredients: list[str]
    ingredient_breakdown: list[IngredientNutrition]
    ingredient_chips: list[IngredientChip]
    estimated_portion_g: float
    nutrition_per_serving: NutritionInfo
    nutrition_per_100g: NutritionInfo
    data_source: str

class RecalculateRequest(BaseModel):
    """Request body for /recalculate — user adds/removes/edits ingredients."""
    ingredients: list[dict]  # [{"name": "rice", "grams": 200}, ...]
    portion_grams: Optional[float] = None       # direct grams override
    serving_size: Optional[str] = None           # "bowl", "plate", "cup", etc.
    num_servings: Optional[float] = None         # 1, 1.5, 2, etc.

class FoodAnalysis(BaseModel):
    success: bool
    is_food: bool = True
    food_name: Optional[str] = None
    confidence: Optional[float] = None
    confidence_warning: Optional[str] = None
    dishes: Optional[list[DishAnalysis]] = None
    # Flat fields for backward compat (first dish)
    ingredients: Optional[list[str]] = None
    ingredient_breakdown: Optional[list[IngredientNutrition]] = None
    ingredient_chips: Optional[list[IngredientChip]] = None
    suggested_ingredients: Optional[list[str]] = None
    estimated_portion_g: Optional[float] = None
    serving_size: Optional[str] = None
    nutrition_per_serving: Optional[NutritionInfo] = None
    nutrition_per_100g: Optional[NutritionInfo] = None
    data_source: Optional[str] = None
    tips: Optional[str] = None
    warnings: Optional[list[str]] = None
    error: Optional[str] = None

# ============== GEMINI: IDENTIFICATION ONLY ==============

IDENTIFY_PROMPT = """Analyze this food image carefully.

FIRST: Determine if this is actually food/beverage. If it's a selfie, screenshot, document, random object, or anything non-food, return:
{"is_food": false, "reason": "brief explanation"}

IF PACKAGED FOOD with a visible nutrition label: Read the label directly and return:
{
    "is_food": true,
    "food_name": "product name from label",
    "confidence": 0.95,
    "is_packaged": true,
    "label_nutrition_per_serving": {
        "serving_size_g": 30,
        "calories": 120,
        "protein_g": 3,
        "carbs_g": 22,
        "fat_g": 2.5,
        "fiber_g": 1
    },
    "tips": "Brief health tip"
}

IF REGULAR FOOD/BEVERAGE: Identify the dish(es) and list individual ingredients with estimated gram quantities.
{
    "is_food": true,
    "food_name": "name of the main dish",
    "confidence": 0.9,
    "dishes": [
        {
            "name": "Chicken Biryani",
            "type": "solid",
            "ingredients": [
                {"name": "basmati rice", "estimated_grams": 200},
                {"name": "chicken", "estimated_grams": 120, "cooking_method": "fried"},
                {"name": "ghee", "estimated_grams": 15}
            ],
            "estimated_portion_g": 335
        }
    ],
    "suggested_missing_ingredients": ["salt", "cumin", "turmeric", "yogurt marinade"],
    "tips": "Brief health tip"
}

Guidelines:
- Be specific about food name (e.g., "Chicken Biryani" not "rice dish") — use natural, human-friendly names
- If there are MULTIPLE separate dishes visible (e.g., rice + curry in separate bowls), list each as a separate dish
- For BEVERAGES, set type to "beverage" and estimate volume in ml (use estimated_grams for ml since density ~1)
- List ALL visible ingredients including cooking fats (oil, ghee, butter)
- INCLUDE the cooking_method field when visible (fried, grilled, steamed, boiled, etc.) — this affects nutrition
- Use simple, common ingredient names (e.g., "chicken" not "grilled boneless chicken thigh")
- Estimate grams based on visual cues (plate size, utensils, hand for scale)
- estimated_portion_g = sum of all ingredient grams
- suggested_missing_ingredients: list ingredients that are LIKELY in the dish but NOT visible (spices, sauces, marinades, hidden fats). These help users add what's missing.
- Works for ALL cuisines worldwide
- Return ONLY valid JSON, no markdown"""

def parse_gemini_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        raise ValueError(f"Could not parse: {text[:200]}")

# ============== INGREDIENT MATCHING ==============

# Cooking methods that significantly affect nutrition (keep for lookup)
COOKING_METHODS = {
    "fried": {"fat_multiplier": 1.5},
    "deep fried": {"fat_multiplier": 2.0},
    "shallow fried": {"fat_multiplier": 1.3},
    "grilled": {},
    "baked": {},
    "roasted": {},
    "steamed": {},
    "boiled": {},
    "raw": {},
    "sauteed": {"fat_multiplier": 1.2},
}

def normalize_ingredient(name: str) -> str:
    """Normalize ingredient name for DB matching. Preserves cooking context."""
    name = name.strip().lower()
    # Only strip non-nutritionally-relevant prefixes
    for prefix in ["fresh ", "dried ", "sliced ", "diced ",
                    "chopped ", "organic ", "whole ", "ground ", "minced "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

def extract_cooking_method(name: str) -> tuple[str, str | None]:
    """Extract cooking method from ingredient name. Returns (clean_name, method)."""
    name_lower = name.strip().lower()
    for method in COOKING_METHODS:
        if name_lower.startswith(method + " "):
            return name_lower[len(method) + 1:], method
    return name_lower, None

def fuzzy_match_db(name: str, db: dict) -> str | None:
    """
    Safe fuzzy matching — avoids false positives like 'rice' matching 'licorice'.
    Uses word-boundary matching instead of raw substring.
    """
    name_words = set(name.split())

    best_key = None
    best_score = 0

    for key in db:
        key_words = set(key.split())

        # Exact word overlap (e.g., "basmati rice" matches "rice")
        overlap = name_words & key_words
        if not overlap:
            continue

        # Score = overlap size / max(name_words, key_words) — penalizes partial matches
        score = len(overlap) / max(len(name_words), len(key_words))

        # Require at least one meaningful word match (skip tiny words like "a", "of")
        meaningful_overlap = {w for w in overlap if len(w) > 2}
        if not meaningful_overlap:
            continue

        if score > best_score:
            best_score = score
            best_key = key

    # Require at least 50% word overlap
    if best_score >= 0.5:
        return best_key
    return None

# ============== USDA API LOOKUP ==============

USDA_CACHE: dict[str, dict | None] = {}

def _check_usda_relevance(query: str, usda_description: str) -> bool:
    """Check that USDA result is actually relevant to what we searched for."""
    query_words = set(normalize_ingredient(query).split())
    desc_words = set(usda_description.lower().replace(",", " ").split())
    # At least one significant query word must appear in the USDA description
    meaningful_query = {w for w in query_words if len(w) > 2}
    if not meaningful_query:
        return True  # very short query, trust USDA
    overlap = meaningful_query & desc_words
    return len(overlap) > 0

async def lookup_usda(ingredient_name: str) -> dict | None:
    """
    Look up nutrition per 100g from USDA FoodData Central.
    Returns dict with calories, protein_g, carbs_g, fat_g, fiber_g per 100g, or None.
    """
    normalized = normalize_ingredient(ingredient_name)

    if normalized in USDA_CACHE:
        return USDA_CACHE[normalized]

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Prefer Foundation and SR Legacy (lab-measured, not branded)
            resp = await client.get(
                "https://api.nal.usda.gov/fdc/v1/foods/search",
                params={
                    "api_key": USDA_API_KEY,
                    "query": normalized,
                    "dataType": "Foundation,SR Legacy",
                    "pageSize": 3,
                },
            )
            if resp.status_code != 200:
                logger.warning(f"USDA API error {resp.status_code} for '{normalized}'")
                USDA_CACHE[normalized] = None
                return None

            data = resp.json()
            foods = data.get("foods", [])
            if not foods:
                # Try broader search including Survey data
                resp = await client.get(
                    "https://api.nal.usda.gov/fdc/v1/foods/search",
                    params={
                        "api_key": USDA_API_KEY,
                        "query": normalized,
                        "pageSize": 3,
                    },
                )
                if resp.status_code != 200:
                    USDA_CACHE[normalized] = None
                    return None
                data = resp.json()
                foods = data.get("foods", [])

            if not foods:
                USDA_CACHE[normalized] = None
                return None

            # Find the first relevant match
            food = None
            for candidate in foods:
                desc = candidate.get("description", "")
                if _check_usda_relevance(normalized, desc):
                    food = candidate
                    break

            if not food:
                logger.info(f"USDA: no relevant match for '{normalized}' (candidates: {[f.get('description','') for f in foods[:3]]})")
                USDA_CACHE[normalized] = None
                return None

            # Extract nutrients from the best match
            nutrients = {}
            for n in food.get("foodNutrients", []):
                nname = n.get("nutrientName", "")
                val = n.get("value", 0)
                if nname == "Energy" and n.get("unitName") == "KCAL":
                    nutrients["calories"] = val
                elif nname == "Protein":
                    nutrients["protein_g"] = val
                elif nname == "Total lipid (fat)":
                    nutrients["fat_g"] = val
                elif nname == "Carbohydrate, by difference":
                    nutrients["carbs_g"] = val
                elif nname == "Fiber, total dietary":
                    nutrients["fiber_g"] = val

            if "calories" not in nutrients:
                USDA_CACHE[normalized] = None
                return None

            result = {
                "calories_per_100g": nutrients.get("calories", 0),
                "protein_per_100g": nutrients.get("protein_g", 0),
                "fat_per_100g": nutrients.get("fat_g", 0),
                "carbs_per_100g": nutrients.get("carbs_g", 0),
                "fiber_per_100g": nutrients.get("fiber_g", 0),
                "usda_food_name": food.get("description", ""),
                "source": "usda",
            }
            elapsed = time.time() - start
            logger.info(f"USDA: '{normalized}' → '{food.get('description','')}' ({elapsed:.1f}s)")
            USDA_CACHE[normalized] = result
            return result

    except Exception as e:
        logger.error(f"USDA lookup failed for '{normalized}': {e}")
        USDA_CACHE[normalized] = None
        return None

# ============== LOCAL DB LOOKUP ==============

def _db_to_per100g(entry: dict, source: str) -> dict:
    """Convert per-gram DB entry to per-100g format."""
    return {
        "calories_per_100g": entry["calories_per_g"] * 100,
        "protein_per_100g": entry["protein_per_g"] * 100,
        "fat_per_100g": entry["fat_per_g"] * 100,
        "carbs_per_100g": entry["carbs_per_g"] * 100,
        "fiber_per_100g": entry["fiber_per_g"] * 100,
        "source": source,
    }

def lookup_local_db(ingredient_name: str) -> dict | None:
    """Look up nutrition from local ingredients_database.csv."""
    normalized = normalize_ingredient(ingredient_name)

    # Exact match
    if normalized in LOCAL_INGREDIENTS_DB:
        return _db_to_per100g(LOCAL_INGREDIENTS_DB[normalized], "local_db")

    # Safe fuzzy match
    match = fuzzy_match_db(normalized, LOCAL_INGREDIENTS_DB)
    if match:
        logger.info(f"Local DB fuzzy: '{normalized}' → '{match}'")
        return _db_to_per100g(LOCAL_INGREDIENTS_DB[match], "local_db")

    # Also try without cooking method
    clean_name, _ = extract_cooking_method(normalized)
    if clean_name != normalized:
        if clean_name in LOCAL_INGREDIENTS_DB:
            return _db_to_per100g(LOCAL_INGREDIENTS_DB[clean_name], "local_db")
        match = fuzzy_match_db(clean_name, LOCAL_INGREDIENTS_DB)
        if match:
            return _db_to_per100g(LOCAL_INGREDIENTS_DB[match], "local_db")

    return None

# ============== IFCT LOOKUP ==============

def lookup_ifct(ingredient_name: str) -> dict | None:
    """Look up nutrition from IFCT Indian food database."""
    normalized = normalize_ingredient(ingredient_name)

    # Exact match
    if normalized in IFCT_DB:
        return _db_to_per100g(IFCT_DB[normalized], "ifct")

    # Safe fuzzy match
    match = fuzzy_match_db(normalized, IFCT_DB)
    if match:
        logger.info(f"IFCT fuzzy: '{normalized}' → '{match}'")
        return _db_to_per100g(IFCT_DB[match], "ifct")

    # Try without cooking method
    clean_name, _ = extract_cooking_method(normalized)
    if clean_name != normalized:
        if clean_name in IFCT_DB:
            return _db_to_per100g(IFCT_DB[clean_name], "ifct")
        match = fuzzy_match_db(clean_name, IFCT_DB)
        if match:
            return _db_to_per100g(IFCT_DB[match], "ifct")

    return None

# ============== COOKING METHOD ADJUSTMENT ==============

def apply_cooking_adjustment(nutrition: dict, cooking_method: str | None) -> dict:
    """Adjust nutrition values based on cooking method (mainly fat from frying)."""
    if not cooking_method or cooking_method not in COOKING_METHODS:
        return nutrition
    adjustments = COOKING_METHODS[cooking_method]
    if "fat_multiplier" in adjustments:
        result = dict(nutrition)
        fat_mult = adjustments["fat_multiplier"]
        added_fat_per_100g = result["fat_per_100g"] * (fat_mult - 1)
        result["fat_per_100g"] = result["fat_per_100g"] * fat_mult
        # Extra fat adds ~9 cal per gram
        result["calories_per_100g"] += added_fat_per_100g * 9
        return result
    return nutrition

# ============== CORE NUTRITION LOOKUP (WATERFALL) ==============

async def get_ingredient_nutrition(name: str, grams: float, cooking_method: str | None = None) -> IngredientNutrition:
    """
    Waterfall lookup for a single ingredient:
    1. IFCT (Indian foods — 126 items, correct context for Indian cuisine)
    2. USDA API (170K+ foods, lab-verified)
    3. Local ingredient DB (551 ingredients)
    4. Gemini estimate (last resort)
    """
    # Clamp grams to sane range
    grams = max(MIN_INGREDIENT_GRAMS, min(grams, MAX_INGREDIENT_GRAMS))
    multiplier = grams / 100.0

    def _build_result(data: dict, source: str) -> IngredientNutrition:
        adjusted = apply_cooking_adjustment(data, cooking_method)
        return IngredientNutrition(
            name=name,
            estimated_grams=grams,
            nutrition=NutritionInfo(
                calories=round(adjusted["calories_per_100g"] * multiplier, 1),
                protein_g=round(adjusted["protein_per_100g"] * multiplier, 1),
                carbs_g=round(adjusted["carbs_per_100g"] * multiplier, 1),
                fat_g=round(adjusted["fat_per_100g"] * multiplier, 1),
                fiber_g=round(adjusted["fiber_per_100g"] * multiplier, 1),
            ),
            source=source,
        )

    # 1. Try IFCT
    ifct = lookup_ifct(name)
    if ifct:
        return _build_result(ifct, "ifct")

    # 2. Try USDA
    usda = await lookup_usda(name)
    if usda:
        return _build_result(usda, "usda")

    # 3. Try local DB
    local = lookup_local_db(name)
    if local:
        return _build_result(local, "local_db")

    # 4. Fallback: ask Gemini for this specific ingredient
    if gemini_client:
        try:
            resp = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=f'Nutrition per 100g of "{name}"{f" ({cooking_method})" if cooking_method else ""}. Return JSON only: {{"calories":N,"protein_g":N,"carbs_g":N,"fat_g":N,"fiber_g":N}}',
            )
            nut = json.loads(re.search(r'\{[^}]+\}', resp.text).group())
            logger.info(f"Gemini fallback for '{name}': {nut}")
            return IngredientNutrition(
                name=name,
                estimated_grams=grams,
                nutrition=NutritionInfo(
                    calories=round(nut.get("calories", 0) * multiplier, 1),
                    protein_g=round(nut.get("protein_g", 0) * multiplier, 1),
                    carbs_g=round(nut.get("carbs_g", 0) * multiplier, 1),
                    fat_g=round(nut.get("fat_g", 0) * multiplier, 1),
                    fiber_g=round(nut.get("fiber_g", 0) * multiplier, 1),
                ),
                source="ai",
            )
        except Exception as e:
            logger.error(f"Gemini nutrition fallback failed for '{name}': {e}")

    # Complete fallback
    return IngredientNutrition(
        name=name,
        estimated_grams=grams,
        nutrition=NutritionInfo(calories=0, protein_g=0, carbs_g=0, fat_g=0, fiber_g=0),
        source="unknown",
    )

# ============== PROCESS A SINGLE DISH ==============

async def process_dish(dish_data: dict, warnings: list[str]) -> DishAnalysis:
    """Process one dish from Gemini's response into full nutrition breakdown."""
    dish_name = dish_data.get("name", "Unknown")
    dish_type = dish_data.get("type", "solid")
    raw_ingredients = dish_data.get("ingredients", [])

    ingredient_breakdown = []
    sources_used = set()

    for ing in raw_ingredients:
        if isinstance(ing, dict):
            name = ing.get("name", "unknown")
            grams = ing.get("estimated_grams", 50)
            cooking_method = ing.get("cooking_method")
        else:
            name = str(ing)
            grams = 50
            cooking_method = None

        # Skip 0g ingredients
        if grams <= 0:
            logger.info(f"Skipping 0g ingredient: {name}")
            continue

        # Warn on suspiciously large portions
        if grams > 1000:
            warnings.append(f"'{name}' estimated at {grams}g seems high — clamped to {MAX_INGREDIENT_GRAMS}g")

        result = await get_ingredient_nutrition(name, grams, cooking_method)
        ingredient_breakdown.append(result)
        sources_used.add(result.source)

    # Calculate totals
    total_cal = sum(i.nutrition.calories for i in ingredient_breakdown)
    total_pro = sum(i.nutrition.protein_g for i in ingredient_breakdown)
    total_carb = sum(i.nutrition.carbs_g for i in ingredient_breakdown)
    total_fat = sum(i.nutrition.fat_g for i in ingredient_breakdown)
    total_fiber = sum(i.nutrition.fiber_g for i in ingredient_breakdown)
    total_grams = sum(i.estimated_grams for i in ingredient_breakdown)

    estimated_portion = total_grams if total_grams > 0 else dish_data.get("estimated_portion_g", 100)

    nutrition_per_serving = NutritionInfo(
        calories=round(total_cal, 1),
        protein_g=round(total_pro, 1),
        carbs_g=round(total_carb, 1),
        fat_g=round(total_fat, 1),
        fiber_g=round(total_fiber, 1),
    )

    per_100 = 100.0 / estimated_portion if estimated_portion > 0 else 1
    nutrition_per_100g = NutritionInfo(
        calories=round(total_cal * per_100, 1),
        protein_g=round(total_pro * per_100, 1),
        carbs_g=round(total_carb * per_100, 1),
        fat_g=round(total_fat * per_100, 1),
        fiber_g=round(total_fiber * per_100, 1),
    )

    # Determine data source
    verified = sources_used & {"usda", "ifct"}
    if verified and sources_used == verified:
        data_source = "verified"
    elif verified:
        data_source = "verified+fallback"
    elif "local_db" in sources_used:
        data_source = "local_db"
    else:
        data_source = "ai"

    # Build simplified chips for UI
    chips = [
        IngredientChip(
            name=i.name.title(),
            cal=round(i.nutrition.calories),
            grams=i.estimated_grams,
        )
        for i in ingredient_breakdown
    ]

    return DishAnalysis(
        dish_name=dish_name,
        dish_type=dish_type,
        ingredients=[i.name for i in ingredient_breakdown],
        ingredient_breakdown=ingredient_breakdown,
        ingredient_chips=chips,
        estimated_portion_g=round(estimated_portion, 1),
        nutrition_per_serving=nutrition_per_serving,
        nutrition_per_100g=nutrition_per_100g,
        data_source=data_source,
    )

# ============== FASTAPI APP ==============

app = FastAPI(title="Food Calorie Tracker", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== ENDPOINTS ==============

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "model": MODEL_NAME,
        "api_configured": gemini_client is not None,
        "usda_api": USDA_API_KEY != "",
        "ifct_ingredients": len(IFCT_DB),
        "local_ingredients": len(LOCAL_INGREDIENTS_DB),
        "serving_sizes": SERVING_SIZES,
    }

@app.post("/analyze", response_model=FoodAnalysis)
async def analyze_food(
    file: UploadFile = File(...),
    serving_size: Optional[str] = Form(None),
    num_servings: Optional[float] = Form(None),
    portion_grams: Optional[float] = Form(None),
):
    if not gemini_client:
        return FoodAnalysis(success=False, error="Gemini API key not configured")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    # --- FIX #1: Image size limit ---
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(413, f"Image too large ({size_mb:.1f}MB). Max is {MAX_IMAGE_SIZE_MB}MB.")

    request_start = time.time()
    warnings: list[str] = []

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # --- FIX #3: Gemini timeout ---
        logger.info(f"Sending image to Gemini ({size_mb:.1f}MB)")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=MODEL_NAME,
                    contents=[IDENTIFY_PROMPT, pil_image],
                ),
                timeout=GEMINI_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(f"Gemini timed out after {GEMINI_TIMEOUT_SECONDS}s")
            return FoodAnalysis(success=False, error=f"AI analysis timed out after {GEMINI_TIMEOUT_SECONDS}s. Try a clearer photo.")

        gemini_time = time.time() - request_start
        logger.info(f"Gemini responded in {gemini_time:.1f}s")

        data = parse_gemini_response(response.text)
        logger.info(f"Gemini raw: {json.dumps(data)[:500]}")

        # --- FIX #2: Non-food detection ---
        if not data.get("is_food", True):
            reason = data.get("reason", "This doesn't appear to be food")
            return FoodAnalysis(success=True, is_food=False, error=reason)

        food_name = data.get("food_name", "Unknown food")
        confidence = data.get("confidence", 0.8)
        tips = data.get("tips", "")

        # --- FIX #7: Confidence threshold ---
        confidence_warning = None
        if confidence < CONFIDENCE_WARN_THRESHOLD:
            confidence_warning = f"Low confidence ({confidence:.0%}). The image may be unclear — results might be inaccurate."
            warnings.append(confidence_warning)
            logger.warning(f"Low confidence {confidence} for '{food_name}'")

        # --- FIX #11: Packaged food with nutrition label ---
        if data.get("is_packaged") and data.get("label_nutrition_per_serving"):
            label = data["label_nutrition_per_serving"]
            serving_g = label.get("serving_size_g", 100)
            nutrition = NutritionInfo(
                calories=label.get("calories", 0),
                protein_g=label.get("protein_g", 0),
                carbs_g=label.get("carbs_g", 0),
                fat_g=label.get("fat_g", 0),
                fiber_g=label.get("fiber_g", 0),
            )
            per_100 = 100.0 / serving_g if serving_g > 0 else 1
            nutrition_per_100g = NutritionInfo(
                calories=round(nutrition.calories * per_100, 1),
                protein_g=round(nutrition.protein_g * per_100, 1),
                carbs_g=round(nutrition.carbs_g * per_100, 1),
                fat_g=round(nutrition.fat_g * per_100, 1),
                fiber_g=round(nutrition.fiber_g * per_100, 1),
            )
            elapsed = time.time() - request_start
            logger.info(f"Packaged food '{food_name}' processed in {elapsed:.1f}s")
            return FoodAnalysis(
                success=True,
                food_name=food_name,
                confidence=confidence,
                estimated_portion_g=serving_g,
                serving_size=f"{serving_g}g (from label)",
                nutrition_per_serving=nutrition,
                nutrition_per_100g=nutrition_per_100g,
                data_source="label",
                tips=tips,
                warnings=warnings if warnings else None,
            )

        # --- FIX #13: Multi-dish support ---
        dishes_data = data.get("dishes", [])
        if not dishes_data:
            # Backward compat: old single-dish format from Gemini
            gemini_ingredients = data.get("ingredients", [])
            dishes_data = [{
                "name": food_name,
                "type": "solid",
                "ingredients": gemini_ingredients,
                "estimated_portion_g": data.get("estimated_portion_g", 100),
            }]

        # --- FIX #14: Detect beverage from name if Gemini didn't flag it ---
        BEVERAGE_KEYWORDS = {"juice", "coffee", "tea", "chai", "lassi", "smoothie", "shake",
                             "milk", "water", "soda", "beer", "wine", "buttermilk", "lemonade",
                             "soup", "rasam", "broth"}
        for dish in dishes_data:
            name_words = set(dish.get("name", "").lower().split())
            if name_words & BEVERAGE_KEYWORDS and dish.get("type") != "beverage":
                dish["type"] = "beverage"

        # Process all dishes
        processed_dishes = []
        for dish_data in dishes_data:
            dish_result = await process_dish(dish_data, warnings)
            processed_dishes.append(dish_result)

        # Extract suggested missing ingredients from Gemini
        suggested_ingredients = data.get("suggested_missing_ingredients", [])

        # Aggregate totals across all dishes for the flat response
        all_ingredients = []
        all_breakdown = []
        all_chips = []
        total_grams = 0
        total_cal = total_pro = total_carb = total_fat = total_fiber = 0
        all_sources = set()

        for d in processed_dishes:
            all_ingredients.extend(d.ingredients)
            all_breakdown.extend(d.ingredient_breakdown)
            all_chips.extend(d.ingredient_chips)
            total_grams += d.estimated_portion_g
            total_cal += d.nutrition_per_serving.calories
            total_pro += d.nutrition_per_serving.protein_g
            total_carb += d.nutrition_per_serving.carbs_g
            total_fat += d.nutrition_per_serving.fat_g
            total_fiber += d.nutrition_per_serving.fiber_g
            all_sources.add(d.data_source)

        estimated_portion = total_grams

        # Apply serving size scaling
        serving_label = None
        scale = 1
        if portion_grams and portion_grams > 0:
            scale = portion_grams / estimated_portion if estimated_portion > 0 else 1
            serving_label = f"{portion_grams}g"
        elif serving_size:
            serving_g = SERVING_SIZES.get(serving_size)
            servings = num_servings or 1
            if serving_g:
                target_g = serving_g * servings
                scale = target_g / estimated_portion if estimated_portion > 0 else 1
                serving_label = f"{servings} {serving_size}" if servings != 1 else serving_size
            else:
                scale = servings
                serving_label = f"{servings} piece" if servings != 1 else "1 piece"
        else:
            serving_label = f"~{round(estimated_portion)}g (estimated)"

        nutrition_per_serving = NutritionInfo(
            calories=round(total_cal * scale, 1),
            protein_g=round(total_pro * scale, 1),
            carbs_g=round(total_carb * scale, 1),
            fat_g=round(total_fat * scale, 1),
            fiber_g=round(total_fiber * scale, 1),
        )

        per_100 = 100.0 / estimated_portion if estimated_portion > 0 else 1
        nutrition_per_100g = NutritionInfo(
            calories=round(total_cal * per_100, 1),
            protein_g=round(total_pro * per_100, 1),
            carbs_g=round(total_carb * per_100, 1),
            fat_g=round(total_fat * per_100, 1),
            fiber_g=round(total_fiber * per_100, 1),
        )

        # Overall data source
        if "verified" in all_sources and len(all_sources) == 1:
            data_source = "verified"
        elif any("verified" in s for s in all_sources):
            data_source = "verified+fallback"
        elif "local_db" in all_sources:
            data_source = "local_db"
        else:
            data_source = "ai"

        elapsed = time.time() - request_start
        logger.info(f"'{food_name}' — {len(all_breakdown)} ingredients, {data_source}, {elapsed:.1f}s total")

        return FoodAnalysis(
            success=True,
            is_food=True,
            food_name=food_name,
            confidence=confidence,
            confidence_warning=confidence_warning,
            dishes=processed_dishes if len(processed_dishes) > 1 else None,
            ingredients=all_ingredients,
            ingredient_breakdown=all_breakdown,
            ingredient_chips=all_chips,
            suggested_ingredients=suggested_ingredients if suggested_ingredients else None,
            estimated_portion_g=round(estimated_portion * scale, 1),
            serving_size=serving_label,
            nutrition_per_serving=nutrition_per_serving,
            nutrition_per_100g=nutrition_per_100g,
            data_source=data_source,
            tips=tips,
            warnings=warnings if warnings else None,
        )

    except Exception as e:
        elapsed = time.time() - request_start
        logger.error(f"Analysis failed after {elapsed:.1f}s: {e}", exc_info=True)
        return FoodAnalysis(success=False, error=str(e))

@app.post("/recalculate")
async def recalculate_nutrition(req: RecalculateRequest):
    """
    Recalculate nutrition when user adds/removes/edits ingredients.
    No image needed — just ingredient list + grams.

    Request: {"ingredients": [{"name": "rice", "grams": 200}, {"name": "chicken", "grams": 120}], "portion_grams": 350}
    """
    if not req.ingredients:
        raise HTTPException(400, "At least one ingredient required")

    start = time.time()
    breakdown = []
    sources_used = set()

    for ing in req.ingredients:
        name = ing.get("name", "unknown")
        grams = ing.get("grams", 50)
        cooking_method = ing.get("cooking_method")
        if grams <= 0:
            continue
        result = await get_ingredient_nutrition(name, grams, cooking_method)
        breakdown.append(result)
        sources_used.add(result.source)

    total_cal = sum(i.nutrition.calories for i in breakdown)
    total_pro = sum(i.nutrition.protein_g for i in breakdown)
    total_carb = sum(i.nutrition.carbs_g for i in breakdown)
    total_fat = sum(i.nutrition.fat_g for i in breakdown)
    total_fiber = sum(i.nutrition.fiber_g for i in breakdown)
    total_grams = sum(i.estimated_grams for i in breakdown)

    # Apply serving size (same logic as /analyze)
    serving_label = None
    scale = 1
    if req.portion_grams and req.portion_grams > 0:
        scale = req.portion_grams / total_grams if total_grams > 0 else 1
        serving_label = f"{req.portion_grams}g"
    elif req.serving_size:
        serving_g = SERVING_SIZES.get(req.serving_size)
        servings = req.num_servings or 1
        if serving_g:
            target_g = serving_g * servings
            scale = target_g / total_grams if total_grams > 0 else 1
            serving_label = f"{servings} {req.serving_size}" if servings != 1 else req.serving_size
        else:
            scale = servings
            serving_label = f"{servings} piece" if servings != 1 else "1 piece"
    else:
        serving_label = f"~{round(total_grams)}g (estimated)"

    portion = total_grams * scale

    chips = [
        IngredientChip(name=i.name.title(), cal=round(i.nutrition.calories * scale), grams=round(i.estimated_grams * scale, 1))
        for i in breakdown
    ]

    per_100 = 100.0 / portion if portion > 0 else 1

    verified = sources_used & {"usda", "ifct"}
    if verified and sources_used == verified:
        data_source = "verified"
    elif verified:
        data_source = "verified+fallback"
    elif "local_db" in sources_used:
        data_source = "local_db"
    else:
        data_source = "ai"

    elapsed = time.time() - start
    logger.info(f"Recalculate: {len(breakdown)} ingredients, {data_source}, {elapsed:.1f}s")

    return {
        "ingredients": [i.name for i in breakdown],
        "ingredient_breakdown": [i.model_dump() for i in breakdown],
        "ingredient_chips": [c.model_dump() for c in chips],
        "estimated_portion_g": round(portion, 1),
        "serving_size": serving_label,
        "nutrition_per_serving": {
            "calories": round(total_cal * scale, 1),
            "protein_g": round(total_pro * scale, 1),
            "carbs_g": round(total_carb * scale, 1),
            "fat_g": round(total_fat * scale, 1),
            "fiber_g": round(total_fiber * scale, 1),
        },
        "nutrition_per_100g": {
            "calories": round(total_cal * per_100, 1),
            "protein_g": round(total_pro * per_100, 1),
            "carbs_g": round(total_carb * per_100, 1),
            "fat_g": round(total_fat * per_100, 1),
            "fiber_g": round(total_fiber * per_100, 1),
        },
        "data_source": data_source,
    }

@app.get("/serving-sizes")
def get_serving_sizes():
    return SERVING_SIZES

# ============== STATIC FILES (for production) ==============

STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
