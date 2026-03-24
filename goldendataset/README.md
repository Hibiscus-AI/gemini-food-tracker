# Golden Dataset

Ground-truth test data for evaluating the Gemini Food Tracker. Based on Google Research's [Nutrition5K](https://github.com/google-research-datasets/Nutrition5k) (CVPR 2021).

## Contents

| File | Rows | Description |
|------|------|-------------|
| `dishes.csv` | 1,000 | One row per dish — total calories, fat, carbs, protein, ingredient list |
| `dish_ingredients.csv` | 5,454 | One row per dish-ingredient pair — per-ingredient nutrition |
| `ingredients_database.csv` | 551 | Ingredient lookup table — calories and macros per gram (USDA sourced) |
| `dish_*/image.jpeg` | 1,000 | One food image per dish |

## Units

All nutrition values use: **kcal** for calories, **grams** for mass/fat/carbs/protein/fiber.

## File Formats

### `dishes.csv`

| Column | Description |
|--------|-------------|
| Dish ID | Links to image folder (e.g., `dish_1561662216/image.jpeg`) |
| Total Mass (grams) | Weight of the dish |
| Total Calories (kcal) | Total energy |
| Total Fat (grams) | Total fat |
| Total Carbs (grams) | Total carbohydrates |
| Total Protein (grams) | Total protein |
| Number of Ingredients | Ingredient count |
| Ingredients | Pipe-separated list (e.g., `pork\|rice\|greens`) |

### `dish_ingredients.csv`

| Column | Description |
|--------|-------------|
| Dish ID | Links to parent dish |
| Ingredient ID | Nutrition5K ingredient ID |
| Ingredient Name | Human-readable name |
| Mass (grams) | Weight of this ingredient |
| Calories (kcal) | Calories from this ingredient |
| Fat / Carbs / Protein (grams) | Macros from this ingredient |

### `ingredients_database.csv`

| Column | Description |
|--------|-------------|
| ingredient | Name (sorted A–Z) |
| calories_per_g | kcal per gram |
| fat_per_g | Fat per gram |
| carbs_per_g | Carbs per gram |
| protein_per_g | Protein per gram |
| fiber_per_g | Fiber per gram |
| source | `usda` or `nutrition5k` (fallback) |

## How to Use

```python
import csv

dish_id = "dish_1561662216"

# Get ground truth
with open("goldendataset/dishes.csv") as f:
    for row in csv.DictReader(f):
        if row["Dish ID"] == dish_id:
            print(f"Calories: {row['Total Calories (kcal)']} kcal")
            print(f"Ingredients: {row['Ingredients']}")
            break

# Image path
image = f"goldendataset/{dish_id}/image.jpeg"
```

## License

Creative Commons V4.0
