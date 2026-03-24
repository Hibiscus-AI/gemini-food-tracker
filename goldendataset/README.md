# Golden Dataset - Nutrition5K

A curated golden test dataset for evaluating the Gemini Food Tracker's ability to predict ingredients and calories from food images.

## Source

Based on Google Research's [Nutrition5K](https://github.com/google-research-datasets/Nutrition5k) dataset (CVPR 2021), containing real cafeteria food plates with verified nutritional information.

## What's Inside

| File / Folder | Description |
|---------------|-------------|
| `dishes.csv` | Summary of all 5,006 dishes with total nutrition and ingredient list |
| `dish_ingredients.csv` | Detailed per-ingredient breakdown (28,455 rows) |
| `ingredients_metadata.csv` | Nutrition info per gram for all 555 unique ingredients |
| `dish_XXXXXXXXXX/` | 4,793 dish folders, each containing multiple food images |

## Units

| Measurement | Unit |
|-------------|------|
| Mass | grams (g) |
| Calories | kilocalories (kcal) |
| Fat, Carbs, Protein | grams (g) |

## How to Read the Dataset

### 1. `dishes.csv`

One row per dish. Each row contains:

```
Dish ID, Total Mass (grams), Total Calories (kcal), Total Fat (grams), Total Carbs (grams), Total Protein (grams), Number of Ingredients, Ingredients
```

- **Dish ID** links to the image folder (e.g., `dish_1561662216` -> `dish_1561662216/frames_sampled30/`)
- **Ingredients** column is a pipe-separated (`|`) list of ingredient names

### 2. `dish_ingredients.csv`

One row per dish-ingredient pair:

```
Dish ID, Ingredient ID, Ingredient Name, Mass (grams), Calories (kcal), Fat (grams), Carbs (grams), Protein (grams)
```

Use this to get the full nutritional breakdown per ingredient for any dish.

### 3. `ingredients_metadata.csv`

Reference table for all 555 unique ingredients with per-gram nutrition values:

```
ingr, id, cal/g, fat(g), carb(g), protein(g)
```

### 4. Image Folders (`dish_XXXXXXXXXX/`)

Each dish folder contains `frames_sampled30/` with images from 4 camera angles:

```
dish_1561662216/
  frames_sampled30/
    camera_A_frame_001.jpeg    # Camera A - side angle
    camera_A_frame_002.jpeg
    camera_B_frame_001.jpeg    # Camera B - side angle
    camera_B_frame_002.jpeg
    camera_C_frame_001.jpeg    # Camera C - side angle
    camera_C_frame_002.jpeg
    camera_D_frame_001.jpeg    # Camera D - side angle
    camera_D_frame_002.jpeg
```

## How to Use for Testing

1. Pick a dish (e.g., `dish_1561662216`)
2. Feed an image from its folder to the Gemini model
3. Get the model's prediction (ingredients + calories)
4. Compare against ground truth in `dishes.csv` / `dish_ingredients.csv`
5. Measure accuracy

### Quick Example (Python)

```python
import csv

# Load ground truth for a dish
dish_id = "dish_1561662216"

with open("dishes.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Dish ID"] == dish_id:
            print(f"Calories: {row['Total Calories (kcal)']} kcal")
            print(f"Ingredients: {row['Ingredients']}")
            break

# Get image path
image_path = f"{dish_id}/frames_sampled30/camera_A_frame_002.jpeg"
```

## Dataset Stats

- **Total dishes**: 5,006
- **Dishes with images**: 4,793
- **Total ingredient entries**: 28,455
- **Unique ingredients**: 555
- **Images per dish**: ~8-20 (multiple angles and frames)

## License

Creative Commons V4.0 - free to use for any purpose, including commercial.
