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

class FoodAnalysis(BaseModel):
    success: bool
    food_name: Optional[str] = None
    confidence: Optional[float] = None
    ingredients: Optional[list[str]] = None
    estimated_portion_g: Optional[float] = None
    nutrition_per_100g: Optional[NutritionInfo] = None
    nutrition_for_portion: Optional[NutritionInfo] = None
    tips: Optional[str] = None
    error: Optional[str] = None

class CalculateRequest(BaseModel):
    food_name: str
    grams: float

# ============== GEMINI PROMPT ==============

FOOD_ANALYSIS_PROMPT = """Analyze this food image and provide detailed nutritional information.

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{
    "food_name": "name of the dish/food",
    "confidence": 0.95,
    "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
    "estimated_portion_g": 250,
    "nutrition_per_100g": {
        "calories": 150,
        "protein_g": 10,
        "carbs_g": 20,
        "fat_g": 5,
        "fiber_g": 2
    },
    "tips": "Brief health tip about this food"
}

Guidelines:
- Be specific about the food name (e.g., "Grilled Chicken Breast with Rice" not just "food")
- Estimate portion size based on visual cues (plate size, utensils, etc.)
- Provide accurate nutrition values per 100g based on standard food databases
- List main visible ingredients
- Confidence should reflect how certain you are (0.0-1.0)

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
        
        # Extract nutrition per 100g
        nut_100g = data.get("nutrition_per_100g", {})
        nutrition_per_100g = NutritionInfo(
            calories=nut_100g.get("calories", 0),
            protein_g=nut_100g.get("protein_g", 0),
            carbs_g=nut_100g.get("carbs_g", 0),
            fat_g=nut_100g.get("fat_g", 0),
            fiber_g=nut_100g.get("fiber_g", 0),
        )
        
        # Calculate for portion
        portion = portion_grams or data.get("estimated_portion_g", 100)
        multiplier = portion / 100.0
        
        nutrition_for_portion = NutritionInfo(
            calories=round(nutrition_per_100g.calories * multiplier, 1),
            protein_g=round(nutrition_per_100g.protein_g * multiplier, 1),
            carbs_g=round(nutrition_per_100g.carbs_g * multiplier, 1),
            fat_g=round(nutrition_per_100g.fat_g * multiplier, 1),
            fiber_g=round(nutrition_per_100g.fiber_g * multiplier, 1),
        )
        
        return FoodAnalysis(
            success=True,
            food_name=data.get("food_name", "Unknown food"),
            confidence=data.get("confidence", 0.8),
            ingredients=data.get("ingredients", []),
            estimated_portion_g=portion,
            nutrition_per_100g=nutrition_per_100g,
            nutrition_for_portion=nutrition_for_portion,
            tips=data.get("tips"),
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
