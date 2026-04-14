# Golden Dataset

Ground-truth test data for evaluating the Gemini Food Tracker. Based on Google Research's [Nutrition5K](https://github.com/google-research-datasets/Nutrition5k) (CVPR 2021).

## Structure

```
goldendataset/
├── metadata/                         # All nutrition data
│   ├── dishes.csv                    # 1,000 dishes — calories, macros, ingredient list
│   ├── dish_ingredients.csv          # Per-ingredient nutrition breakdown (5,454 rows)
│   └── ingredients_database.csv      # 551 ingredients with USDA nutrition values
├── dish_*/image.jpeg                 # 1,000 food images (one per dish)
└── README.md
```

## Metadata

### `metadata/dishes.csv`

One row per dish. Links to images via Dish ID.

| Column | Description |
|--------|-------------|
| Dish ID | Links to image folder (e.g., `dish_1561662216/image.jpeg`) |
| Total Mass (grams) | Weight of the dish |
| Total Calories (kcal) | Total energy |
| Total Fat / Carbs / Protein (grams) | Macronutrients |
| Number of Ingredients | Ingredient count |
| Ingredients | Pipe-separated list (e.g., `pork\|rice\|greens`) |

### `metadata/dish_ingredients.csv`

One row per dish-ingredient pair (5,454 rows).

| Column | Description |
|--------|-------------|
| Dish ID | Links to parent dish |
| Ingredient ID | Nutrition5K ingredient ID |
| Ingredient Name | Human-readable name |
| Mass / Calories / Fat / Carbs / Protein | Nutrition from this ingredient |

### `metadata/ingredients_database.csv`

Single lookup table for all 551 ingredients. Values are per gram. Sourced from USDA FoodData Central.

| Column | Description |
|--------|-------------|
| ingredient | Name (sorted A–Z) |
| calories_per_g | kcal per gram |
| fat_per_g / carbs_per_g / protein_per_g / fiber_per_g | Macros per gram |
| source | `usda` or `nutrition5k` (fallback) |

## Units

All values: **kcal** for calories, **grams** for everything else.

## License

Creative Commons V4.0
