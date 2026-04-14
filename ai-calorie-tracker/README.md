---
title: AI Calorie Tracker
emoji: 🍕
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🍕 AI Calorie Tracker (Gemini)

**Food analysis using Google's Gemini AI - No training required!**

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🍕 **Food Recognition** | Identifies food from photos |
| 🥗 **Ingredients** | Detects visible ingredients |
| 📊 **Nutrition** | Calories, protein, carbs, fat, fiber |
| ⚖️ **Portion Estimation** | AI estimates portion size |
| 💰 **Powered by Gemini** | AI-assisted analysis |

## 🚀 Quick Start (5 minutes!)

### Step 1: Get Gemini API Key

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with Google
3. Click "Create API Key"
4. Copy your key

### Step 2: Setup Backend

```bash
cd gemini-food-tracker/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your_api_key_here"
# Windows: set GEMINI_API_KEY=your_api_key_here

# Run server
python main.py
# or: uvicorn main:app --reload --port 8000
```

### Step 3: Setup Frontend

```bash
cd gemini-food-tracker/frontend
npm install
npm run dev
```

### Step 4: Open App

Visit: **http://localhost:5173**

## 📸 How It Works

```
┌─────────────────────────────────────────┐
│  1. Upload food photo                   │
│     📸                                  │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. Gemini AI analyzes image            │
│     🤖 "Grilled Chicken with Rice"      │
│     🥗 Ingredients: chicken, rice...    │
│     ⚖️ Estimated: 250g portion          │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. Adjust portion if needed            │
│     [50g] [100g] [150g] [200g] [250g]  │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  4. Get nutrition results!              │
│     🔥 450 calories                     │
│     🥩 35g protein                      │
│     🍞 40g carbs                        │
│     🧈 12g fat                          │
└─────────────────────────────────────────┘
```

## 💰 Pricing

See Gemini pricing and rate limits in your Google account.

## 🔌 API Endpoints

### `POST /analyze`
Upload image, get full analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@food.jpg"
```

Response:
```json
{
  "success": true,
  "food_name": "Grilled Chicken Breast with Rice",
  "confidence": 0.95,
  "ingredients": ["chicken breast", "rice", "broccoli", "olive oil"],
  "estimated_portion_g": 250,
  "nutrition_per_100g": {
    "calories": 150,
    "protein_g": 20,
    "carbs_g": 15,
    "fat_g": 4,
    "fiber_g": 2
  },
  "nutrition_for_portion": {
    "calories": 375,
    "protein_g": 50,
    "carbs_g": 37.5,
    "fat_g": 10,
    "fiber_g": 5
  },
  "tips": "Great protein source! Consider adding more vegetables for fiber."
}
```

### `POST /analyze-with-grams`
Specify exact portion size

```bash
curl -X POST "http://localhost:8000/analyze-with-grams" \
  -F "file=@food.jpg" \
  -F "grams=200"
```

### `POST /calculate`
Get nutrition for food by name

```bash
curl -X POST "http://localhost:8000/calculate" \
  -H "Content-Type: application/json" \
  -d '{"food_name": "banana", "grams": 120}'
```

### `GET /health`
Check API status

### `GET /usage`
Get usage information

## 🆚 Comparison: Gemini vs Training Your Own Model

| Aspect | Gemini API | Custom Model (MM-Food-100K) |
|--------|------------|----------------------------|
| Setup time | 5 minutes | Hours |
| Training | None | Required |
| Cost | See Gemini pricing | Free (self-hosted) |
| Accuracy | ~90%+ | ~70-85% |
| Ingredients | ✅ Yes | ❌ No |
| Portion estimation | ✅ Yes | ❌ No |
| Offline | ❌ No | ✅ Yes |
| Latency | ~1-3s | ~50ms |

## 🔧 Troubleshooting

### "API key not configured"
```bash
export GEMINI_API_KEY="your_key_here"
```

### "Rate limit exceeded"
- Check Gemini pricing and rate limits in your Google account
- Wait a minute or upgrade

### Slow responses
- Normal: 1-3 seconds for image analysis
- First request may be slower

## 📁 Project Structure

```
gemini-food-tracker/
├── backend/
│   ├── main.py           # FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx       # React app
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   └── index.html
└── README.md
```

## 🚀 Deploy to Production

### Option 1: Railway/Render (Easy)
```bash
# Backend deploys automatically from GitHub
# Set GEMINI_API_KEY in environment variables
```

### Option 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📱 Integration Example

```python
import requests

def analyze_food(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/analyze',
            files={'file': f}
        )
    return response.json()

# Usage
result = analyze_food('lunch.jpg')
print(f"Food: {result['food_name']}")
print(f"Calories: {result['nutrition_for_portion']['calories']}")
```

## 📄 License

MIT - Use freely!
