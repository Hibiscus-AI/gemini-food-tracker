# Gemini Food Tracker

AI-powered food analysis using Google Gemini. Upload a photo, get ingredients, calories, and macros — no model training required.

## Overview

| | |
|---|---|
| **What it does** | Identifies food, detects ingredients, estimates calories and macros from a photo |
| **Powered by** | Google Gemini 2.0 Flash |
| **Stack** | FastAPI (Python) + React (TypeScript) |
| **Golden Dataset** | 1,000 dishes with verified nutrition data + 551 ingredient database |

## Setup

**Prerequisites:** Python 3.11+, Node.js 18+, [Gemini API Key](https://makersuite.google.com/app/apikey)

```bash
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
python main.py

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

Open **http://localhost:5173**

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Upload image, get full nutrition analysis |
| `/analyze-with-grams` | POST | Analyze with specified portion size |
| `/calculate` | POST | Get nutrition by food name + weight |
| `/health` | GET | API status check |

## Project Structure

```
gemini-food-tracker/
├── backend/                        # FastAPI server
│   ├── main.py                     # API endpoints + Gemini integration
│   └── requirements.txt
├── frontend/                       # React app
│   ├── src/
│   └── package.json
├── goldendataset/                  # Ground-truth test data
│   ├── metadata/                   # Nutrition CSVs
│   │   ├── dishes.csv              # 1,000 dishes — calories, macros, ingredients
│   │   ├── dish_ingredients.csv    # Per-ingredient nutrition breakdown
│   │   └── ingredients_database.csv # 551 ingredients (USDA)
│   ├── dish_*/image.jpeg           # Food images (1 per dish)
│   └── README.md                   # Dataset documentation
└── Dockerfile
```

## Golden Dataset

The `goldendataset/` directory contains 1,000 real cafeteria dishes from Google's Nutrition5K research, each with a food image and verified nutrition data. Used for evaluating model accuracy.

See [`goldendataset/README.md`](goldendataset/README.md) for details.

## Deploy

```bash
docker build -t gemini-food-tracker .
docker run -e GEMINI_API_KEY="your_key" -p 8000:8000 gemini-food-tracker
```

Or deploy via Railway / Render with `GEMINI_API_KEY` set in environment variables.

## License

MIT
