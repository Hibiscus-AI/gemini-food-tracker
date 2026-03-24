"""
Food Calorie Tracker API - Powered by Gemini

Features:
- Food recognition from photo
- Calorie & macro estimation
- Ingredient detection
- Portion size estimation
Get your API key: https://makersuite.google.com/app/apikey
"""

import os
import csv
import json
import base64
import re
from io import BytesIO
from typing import Optional

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from google import genai

# ============== CONFIGURATION ==============

# Get API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set!")
    print("   Get an API key: https://makersuite.google.com/app/apikey")
    print("   Then: export GEMINI_API_KEY=your_key_here")

# Configure Gemini
MODEL_NAME = "gemini-2.0-flash"
if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
    print("✅ Gemini 2.0 Flash configured")
    print(f"   Model: {MODEL_NAME}")
else:
    client = None

# ============== GOLDEN DATASET ==============

DATASET_DIR = Path(__file__).parent.parent.parent / "goldendataset" / "metadata"

def load_ingredients_database():
    """Load ingredients_database.csv into a dict: ingredient_name -> nutrition per gram."""
    db = {}
    csv_path = DATASET_DIR / "ingredients_database.csv"
    if not csv_path.exists():
        print(f"⚠️  ingredients_database.csv not found at {csv_path}")
        return db
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["ingredient"].strip().lower()
            db[name] = {
                "calories_per_g": float(row["calories_per_g"]),
                "fat_per_g": float(row["fat_per_g"]),
                "carbs_per_g": float(row["carbs_per_g"]),
                "protein_per_g": float(row["protein_per_g"]),
                "fiber_per_g": float(row["fiber_per_g"]),
            }
    print(f"✅ Loaded {len(db)} ingredients from database")
    return db

def load_dishes():
    """Load dishes.csv into a dict keyed by sorted ingredient set."""
    dishes = {}
    csv_path = DATASET_DIR / "dishes.csv"
    if not csv_path.exists():
        print(f"⚠️  dishes.csv not found at {csv_path}")
        return dishes
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row["Dish ID"].strip()
            ingredients_str = row["Ingredients"].strip().lower()
            ing_set = frozenset(i.strip() for i in ingredients_str.split("|"))
            dishes[dish_id] = {
                "ingredients": ing_set,
                "total_mass_g": float(row["Total Mass (grams)"]),
                "calories": float(row["Total Calories (kcal)"]),
                "fat_g": float(row["Total Fat (grams)"]),
                "carbs_g": float(row["Total Carbs (grams)"]),
                "protein_g": float(row["Total Protein (grams)"]),
            }
    print(f"✅ Loaded {len(dishes)} dishes from golden dataset")
    return dishes

def load_dish_ingredients():
    """Load dish_ingredients.csv into a dict: dish_id -> list of ingredient details."""
    dish_ings = {}
    csv_path = DATASET_DIR / "dish_ingredients.csv"
    if not csv_path.exists():
        return dish_ings
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row["Dish ID"].strip()
            if dish_id not in dish_ings:
                dish_ings[dish_id] = []
            dish_ings[dish_id].append({
                "name": row["Ingredient Name"].strip().lower(),
                "mass_g": float(row["Mass (grams)"]),
                "calories": float(row["Calories (kcal)"]),
                "fat_g": float(row["Fat (grams)"]),
                "carbs_g": float(row["Carbs (grams)"]),
                "protein_g": float(row["Protein (grams)"]),
            })
    print(f"✅ Loaded ingredient details for {len(dish_ings)} dishes")
    return dish_ings

def match_dish_by_ingredients(detected_ingredients: list[str], dishes_db: dict, dish_ingredients_db: dict):
    """
    Fuzzy match detected ingredients against golden dataset dishes.
    Returns (dish_id, match_score) or None.
    """
    detected = set(normalize_ingredient(i) for i in detected_ingredients)
    if not detected:
        return None

    best_dish_id = None
    best_score = 0.0

    for dish_id, dish_data in dishes_db.items():
        golden_ings = dish_data["ingredients"]
        if not golden_ings:
            continue

        # Fuzzy ingredient matching: for each detected ingredient,
        # check if it matches any golden ingredient via substring
        matched = 0
        for det in detected:
            for gold in golden_ings:
                if det == gold or det in gold or gold in det:
                    matched += 1
                    break
        # Also check reverse: golden ingredients found in detected
        reverse_matched = 0
        for gold in golden_ings:
            for det in detected:
                if det == gold or det in gold or gold in det:
                    reverse_matched += 1
                    break

        # Use F1-style score (harmonic mean of precision and recall)
        if matched == 0 and reverse_matched == 0:
            continue
        precision = matched / len(detected) if detected else 0
        recall = reverse_matched / len(golden_ings) if golden_ings else 0
        if precision + recall == 0:
            continue
        score = 2 * (precision * recall) / (precision + recall)

        if score > best_score:
            best_score = score
            best_dish_id = dish_id

    # Require at least 50% overlap
    if best_score >= 0.5 and best_dish_id:
        return best_dish_id, best_score
    return None

def calibrate_with_golden(detected_ingredients: list[str], gemini_portion_g: float,
                          dishes_db: dict, dish_ingredients_db: dict, ingredients_db: dict):
    """
    Try to match detected ingredients to golden dataset.
    If matched, use golden dataset nutrition (scaled to Gemini's estimated portion).
    Returns (ingredient_breakdown, source) or (None, None).
    """
    match = match_dish_by_ingredients(detected_ingredients, dishes_db, dish_ingredients_db)
    if not match:
        return None, None

    dish_id, score = match
    dish_data = dishes_db[dish_id]
    golden_ings = dish_ingredients_db.get(dish_id, [])

    if not golden_ings:
        return None, None

    golden_total_mass = dish_data["total_mass_g"]
    if golden_total_mass <= 0:
        return None, None

    # Scale golden dataset nutrition to match Gemini's estimated portion
    scale = gemini_portion_g / golden_total_mass

    breakdown = []
    for ing in golden_ings:
        # Look up fiber from ingredients_database (dish_ingredients.csv doesn't have fiber)
        fiber = 0.0
        ing_db_entry = ingredients_db.get(ing["name"]) or fuzzy_match_ingredient(ing["name"], ingredients_db)
        if ing_db_entry:
            fiber = ing_db_entry["fiber_per_g"] * ing["mass_g"] * scale

        breakdown.append(IngredientNutrition(
            name=ing["name"],
            estimated_grams=round(ing["mass_g"] * scale, 1),
            nutrition=NutritionInfo(
                calories=round(ing["calories"] * scale, 1),
                protein_g=round(ing["protein_g"] * scale, 1),
                carbs_g=round(ing["carbs_g"] * scale, 1),
                fat_g=round(ing["fat_g"] * scale, 1),
                fiber_g=round(fiber, 1),
            ),
        ))

    return breakdown, round(score * 100)

# Load datasets at startup
INGREDIENTS_DB = load_ingredients_database()
DISHES_DB = load_dishes()
DISH_INGREDIENTS_DB = load_dish_ingredients()

# ============== FASTAPI APP ==============

app = FastAPI(
    title="Food Calorie Tracker (Gemini)",
    description="Food analysis powered by Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class FoodAnalysis(BaseModel):
    success: bool
    food_name: Optional[str] = None
    confidence: Optional[float] = None
    ingredients: Optional[list[str]] = None
    ingredient_breakdown: Optional[list[IngredientNutrition]] = None
    estimated_portion_g: Optional[float] = None
    nutrition_per_100g: Optional[NutritionInfo] = None
    nutrition_for_portion: Optional[NutritionInfo] = None
    tips: Optional[str] = None
    data_source: Optional[str] = None  # "verified", "hybrid", or "ai"
    error: Optional[str] = None

class CalculateRequest(BaseModel):
    food_name: str
    grams: float

# ============== GEMINI PROMPT ==============

FOOD_ANALYSIS_PROMPT = """Analyze this food image. Identify the dish and break it down into individual ingredients with estimated quantities and nutrition for each.

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{
    "food_name": "name of the dish/food",
    "confidence": 0.95,
    "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
    "ingredient_breakdown": [
        {
            "name": "rice",
            "estimated_grams": 200,
            "nutrition": {
                "calories": 260,
                "protein_g": 5.4,
                "carbs_g": 56,
                "fat_g": 0.6,
                "fiber_g": 0.8
            }
        },
        {
            "name": "chicken curry",
            "estimated_grams": 150,
            "nutrition": {
                "calories": 225,
                "protein_g": 27,
                "carbs_g": 6,
                "fat_g": 10.5,
                "fiber_g": 1.2
            }
        }
    ],
    "estimated_portion_g": 350,
    "tips": "Brief health tip about this food"
}

Guidelines:
- Be specific about the food name (e.g., "Chicken Biryani" not just "rice dish")
- Break down into individual ingredients/components (rice, dal, ghee, vegetables, meat, oil, etc.)
- For each ingredient, estimate the grams visible on the plate and provide its TOTAL nutrition for that amount (NOT per 100g)
- Use accurate nutrition values based on standard food composition data
- Include cooking fats (oil, ghee, butter) as separate ingredients — they significantly affect macros
- estimated_portion_g should be the sum of all ingredient grams
- Estimate portion size based on visual cues (plate size, utensils, hand, etc.)
- Confidence should reflect how certain you are (0.0-1.0)
- This must work for ALL cuisines — Indian, Asian, Mediterranean, etc.

Return ONLY valid JSON, no explanation or markdown."""

# ============== HELPER FUNCTIONS ==============

def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini response, handling various formats."""
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        raise ValueError(f"Could not parse response: {text[:200]}")

def fuzzy_match_ingredient(name: str, db: dict) -> dict | None:
    """Find the best matching ingredient in the database using substring/token matching."""
    name = name.strip().lower()
    if name in db:
        return db[name]
    # Check substring matches, prefer longest (most specific)
    best_key = None
    best_len = 0
    for key in db:
        if key in name or name in key:
            if len(key) > best_len:
                best_key = key
                best_len = len(key)
    if best_key:
        return db[best_key]
    # Token overlap
    for token in name.split():
        if token in db:
            return db[token]
    return None

def normalize_ingredient(name: str) -> str:
    """Normalize ingredient name for matching."""
    name = name.strip().lower()
    # Remove common prefixes that don't change the base ingredient
    prefixes = ["grilled ", "fried ", "baked ", "roasted ", "steamed ", "boiled ",
                 "raw ", "fresh ", "dried ", "sliced ", "diced ", "chopped ",
                 "cooked ", "organic ", "whole "]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
    return name

def image_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")

# ============== ENDPOINTS ==============

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_configured": client is not None,
        "golden_dataset": {
            "ingredients": len(INGREDIENTS_DB),
            "dishes": len(DISHES_DB),
        },
        "usage_limits_note": "Check Gemini pricing and rate limits."
    }

@app.post("/analyze", response_model=FoodAnalysis)
async def analyze_food(
    file: UploadFile = File(...),
    portion_grams: Optional[float] = Form(None)
):
    """
    Analyze food image and get nutritional information.
    
    - Upload a food photo
    - Get: food name, calories, protein, carbs, fat, ingredients
    - Optionally specify portion size in grams
    """
    if not client:
        return FoodAnalysis(
            success=False,
            error="Gemini API key not configured. Get an API key at: https://makersuite.google.com/app/apikey"
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")
    
    try:
        # Read and prepare image
        image_bytes = await file.read()
        pil_image = image_to_pil(image_bytes)
        
        # Call Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[FOOD_ANALYSIS_PROMPT, pil_image],
        )
        
        # Parse response
        data = parse_gemini_response(response.text)
        detected_ingredients = data.get("ingredients", [])
        gemini_portion = data.get("estimated_portion_g", 100)
        data_source = "ai"

        # Try golden dataset calibration first
        golden_breakdown, match_score = calibrate_with_golden(
            detected_ingredients, gemini_portion,
            DISHES_DB, DISH_INGREDIENTS_DB, INGREDIENTS_DB,
        )

        if golden_breakdown:
            ingredient_breakdown = golden_breakdown
            data_source = "verified"
            tips = (data.get("tips", "") or "") + f" [Nutrition from verified dataset — {match_score}% ingredient match]"
        else:
            # Fall back to Gemini estimates, but calibrate per-ingredient using ingredients_database
            ingredient_breakdown = []
            for ing in data.get("ingredient_breakdown", []):
                ing_name = ing.get("name", "unknown").strip().lower()
                ing_grams = ing.get("estimated_grams", 0)
                normalized = normalize_ingredient(ing_name)
                db_entry = INGREDIENTS_DB.get(normalized) or fuzzy_match_ingredient(normalized, INGREDIENTS_DB)

                if db_entry and ing_grams > 0:
                    # Use verified per-gram nutrition from database
                    ing_nutrition = NutritionInfo(
                        calories=round(db_entry["calories_per_g"] * ing_grams, 1),
                        protein_g=round(db_entry["protein_per_g"] * ing_grams, 1),
                        carbs_g=round(db_entry["carbs_per_g"] * ing_grams, 1),
                        fat_g=round(db_entry["fat_per_g"] * ing_grams, 1),
                        fiber_g=round(db_entry["fiber_per_g"] * ing_grams, 1),
                    )
                    data_source = "hybrid"
                else:
                    # Pure Gemini estimate for unknown ingredients
                    nut = ing.get("nutrition", {})
                    ing_nutrition = NutritionInfo(
                        calories=nut.get("calories", 0),
                        protein_g=nut.get("protein_g", 0),
                        carbs_g=nut.get("carbs_g", 0),
                        fat_g=nut.get("fat_g", 0),
                        fiber_g=nut.get("fiber_g", 0),
                    )

                ingredient_breakdown.append(IngredientNutrition(
                    name=ing_name,
                    estimated_grams=ing_grams,
                    nutrition=ing_nutrition,
                ))
            tips = data.get("tips", "")
            if data_source == "hybrid":
                tips = (tips or "") + " [Macros calibrated with verified ingredient database]"

        # Sum totals from breakdown
        total_calories = sum(i.nutrition.calories for i in ingredient_breakdown)
        total_protein = sum(i.nutrition.protein_g for i in ingredient_breakdown)
        total_carbs = sum(i.nutrition.carbs_g for i in ingredient_breakdown)
        total_fat = sum(i.nutrition.fat_g for i in ingredient_breakdown)
        total_fiber = sum(i.nutrition.fiber_g for i in ingredient_breakdown)
        total_grams = sum(i.estimated_grams for i in ingredient_breakdown)

        estimated_portion = total_grams if total_grams > 0 else gemini_portion
        portion = portion_grams or estimated_portion

        # Nutrition for the full plate
        nutrition_for_portion_full = NutritionInfo(
            calories=round(total_calories, 1),
            protein_g=round(total_protein, 1),
            carbs_g=round(total_carbs, 1),
            fat_g=round(total_fat, 1),
            fiber_g=round(total_fiber, 1),
        )

        # Derive per-100g
        per_100_mult = 100.0 / estimated_portion if estimated_portion > 0 else 1.0
        nutrition_per_100g = NutritionInfo(
            calories=round(total_calories * per_100_mult, 1),
            protein_g=round(total_protein * per_100_mult, 1),
            carbs_g=round(total_carbs * per_100_mult, 1),
            fat_g=round(total_fat * per_100_mult, 1),
            fiber_g=round(total_fiber * per_100_mult, 1),
        )

        # If user specified custom portion, scale from per-100g
        if portion_grams:
            mult = portion_grams / 100.0
            nutrition_for_portion = NutritionInfo(
                calories=round(nutrition_per_100g.calories * mult, 1),
                protein_g=round(nutrition_per_100g.protein_g * mult, 1),
                carbs_g=round(nutrition_per_100g.carbs_g * mult, 1),
                fat_g=round(nutrition_per_100g.fat_g * mult, 1),
                fiber_g=round(nutrition_per_100g.fiber_g * mult, 1),
            )
        else:
            nutrition_for_portion = nutrition_for_portion_full

        return FoodAnalysis(
            success=True,
            food_name=data.get("food_name", "Unknown food"),
            confidence=data.get("confidence", 0.8),
            ingredients=detected_ingredients,
            ingredient_breakdown=ingredient_breakdown,
            estimated_portion_g=portion,
            nutrition_per_100g=nutrition_per_100g,
            nutrition_for_portion=nutrition_for_portion,
            tips=tips,
            data_source=data_source,
        )
        
    except Exception as e:
        return FoodAnalysis(
            success=False,
            error=str(e)
        )

@app.post("/analyze-with-grams", response_model=FoodAnalysis)
async def analyze_with_grams(
    file: UploadFile = File(...),
    grams: float = Form(...)
):
    """
    Analyze food and calculate nutrition for specific portion size.
    """
    return await analyze_food(file, portion_grams=grams)

@app.post("/calculate")
async def calculate_nutrition(request: CalculateRequest):
    """
    Calculate nutrition for a food by name and grams.
    Uses Gemini to get nutrition data.
    """
    if not client:
        return {"success": False, "error": "API not configured"}
    
    prompt = f"""Provide nutrition information for {request.food_name}.

Return JSON only:
{{
    "food_name": "{request.food_name}",
    "nutrition_per_100g": {{
        "calories": number,
        "protein_g": number,
        "carbs_g": number,
        "fat_g": number,
        "fiber_g": number
    }}
}}"""
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        data = parse_gemini_response(response.text)
        
        nut_100g = data.get("nutrition_per_100g", {})
        multiplier = request.grams / 100.0
        
        return {
            "success": True,
            "food_name": request.food_name,
            "grams": request.grams,
            "nutrition_per_100g": nut_100g,
            "nutrition_for_portion": {
                "calories": round(nut_100g.get("calories", 0) * multiplier, 1),
                "protein_g": round(nut_100g.get("protein_g", 0) * multiplier, 1),
                "carbs_g": round(nut_100g.get("carbs_g", 0) * multiplier, 1),
                "fat_g": round(nut_100g.get("fat_g", 0) * multiplier, 1),
                "fiber_g": round(nut_100g.get("fiber_g", 0) * multiplier, 1),
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/usage")
def get_usage_info():
    """Get information about API usage limits."""
    return {
        "model": "Gemini 2.0 Flash",
        "usage_limits_note": "Check Gemini pricing and rate limits.",
        "get_api_key": "https://makersuite.google.com/app/apikey",
    }

# ============== STATIC FILES (for production) ==============

STATIC_DIR = Path(__file__).parent / "static"

# Serve static files if the directory exists (production build)
if STATIC_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    # Serve static assets
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Catch-all for SPA routing
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = STATIC_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
