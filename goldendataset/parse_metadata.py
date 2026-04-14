import csv
import os

BASE = "/Users/karunyamunagala/Desktop/gemini-food-tracker/goldendataset"
INPUT_FILES = [os.path.join(BASE, f) for f in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]]

dishes = []
ingredients = []

for path in INPUT_FILES:
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        continue
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            vals = [v.strip() for v in row]
            dish_id = vals[0]
            total_mass = vals[1]
            total_cal = vals[2]
            total_fat = vals[3]
            total_carbs = vals[4]
            total_protein = vals[5]

            # Remaining fields are groups of 7: id, name, mass, cal, fat, carbs, protein
            ingr_data = vals[6:]
            ingr_names = []
            num_ingr = 0
            for i in range(0, len(ingr_data), 7):
                if i + 6 >= len(ingr_data) + 1:
                    break
                chunk = ingr_data[i:i+7]
                if len(chunk) < 7:
                    break
                ingr_id, ingr_name, mass, cal, fat, carbs, protein = chunk
                ingr_names.append(ingr_name)
                num_ingr += 1
                ingredients.append([dish_id, ingr_id, ingr_name, mass, cal, fat, carbs, protein])

            dishes.append([dish_id, total_mass, total_cal, total_fat, total_carbs, total_protein,
                           num_ingr, "|".join(ingr_names)])

# Write dishes.csv
with open(os.path.join(BASE, "dishes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dish_id","total_mass_g","total_calories","total_fat_g","total_carbs_g","total_protein_g","num_ingredients","ingredient_names"])
    w.writerows(dishes)

# Write dish_ingredients.csv
with open(os.path.join(BASE, "dish_ingredients.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dish_id","ingredient_id","ingredient_name","mass_g","calories","fat_g","carbs_g","protein_g"])
    w.writerows(ingredients)

print(f"Done: {len(dishes)} dishes, {len(ingredients)} ingredient rows")
