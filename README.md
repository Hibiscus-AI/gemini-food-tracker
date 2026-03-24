# Gemini Food Tracker

AI-powered food analysis using Google Gemini. Upload a photo, get ingredients, calories, and macros — no model training required.

## Overview

| | |
|---|---|
| **What it does** | Identifies food, detects ingredients, estimates calories and macros from a photo |
| **Powered by** | Google Gemini API |
| **Stack** | FastAPI (Python) + React (TypeScript) |
| **Golden Dataset** | Nutrition5K — 5,006 dishes with verified nutrition data |

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
├── backend/              # FastAPI server
├── frontend/             # React app
├── goldendataset/        # Nutrition5K test data (5,006 dishes)
│   ├── dishes.csv        # Dish-level nutrition summary
│   ├── dish_ingredients.csv  # Per-ingredient breakdown
│   ├── ingredients_metadata.csv
│   └── README.md         # Dataset documentation
└── Dockerfile
```

## Golden Dataset

The `goldendataset/` contains Nutrition5K metadata for evaluation. Dish images (6 GB) are stored separately on [Google Drive]().

See [`goldendataset/README.md`](goldendataset/README.md) for full documentation.

## Deploy

```bash
# Docker
docker build -t gemini-food-tracker .
docker run -e GEMINI_API_KEY="your_key" -p 8000:8000 gemini-food-tracker
```

Or deploy via Railway / Render with `GEMINI_API_KEY` set in environment variables.

## License

MIT
