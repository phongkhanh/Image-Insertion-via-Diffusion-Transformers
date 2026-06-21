import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

parquet_dir = "/data1/stage/navsim_workspace/AnyInsertion/text_prompt"
output_dir = "/data1/stage/navsim_workspace/AnyInsertion/data_text"

os.makedirs(output_dir, exist_ok=True)

category_label_maps = {}

print("Processing all parquet files...")

# Get all parquet files
parquet_files = sorted([
    f for f in os.listdir(parquet_dir)
    if f.endswith(".parquet")
])

for filename in parquet_files:

    parquet_path = os.path.join(parquet_dir, filename)

    print(f"\nLoading: {filename}")

    df = pd.read_parquet(parquet_path)

    print("Columns:")
    print(df.columns.tolist())

    # Automatically detect image/mask columns
    image_fields = [
        col for col in df.columns
        if ("image" in col.lower()) or ("mask" in col.lower())
    ]

    print("Detected image fields:")
    print(image_fields)

    for i, row in tqdm(df.iterrows(), total=len(df)):

        try:

            split = str(row.get("split", "unknown"))
            category = str(row.get("category", "unknown"))

            label = str(row.get("label", "unknown"))
            src_label = str(row.get("src_label", "unknown"))
            src_type = str(row.get("src_type", "unknown"))

            # Build pseudo text prompt
            prompt = f"replace {src_label} with {label}"

            sample_id_full = str(row.get("id", f"{i:06}"))
            sample_id = sample_id_full.split("/")[-1]

            image_name = f"{sample_id}.png"

            key = (split, category)

            if key not in category_label_maps:
                category_label_maps[key] = {}

            # Save all image/mask fields
            for field in image_fields:

                try:

                    img_dict = row.get(field, None)

                    if img_dict is None:
                        continue

                    if not isinstance(img_dict, dict):
                        continue

                    if "bytes" not in img_dict:
                        continue

                    image_bytes = img_dict["bytes"]

                    image_np = cv2.imdecode(
                        np.frombuffer(image_bytes, np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )

                    if image_np is None:
                        print(f"Failed decode: {field} | {image_name}")
                        continue

                    # Ensure mask is uint8
                    if "mask" in field.lower():
                        if image_np.dtype != np.uint8:
                            image_np = image_np.astype(np.uint8)

                    save_dir = os.path.join(
                        output_dir,
                        split,
                        category,
                        field
                    )

                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, image_name)

                    cv2.imwrite(save_path, image_np)

                except Exception as e:
                    print(f"[ERROR] {filename} | {field} | {image_name}")
                    print(e)

            # Save metadata
            category_label_maps[key][image_name] = {
                "label": label,
                "src_label": src_label,
                "src_type": src_type,
                "prompt": prompt
            }

        except Exception as e:
            print(f"[ROW ERROR] {filename} | row {i}")
            print(e)

# Save label json files
print("\nSaving label files...")

for (split, category), label_dict in category_label_maps.items():

    save_dir = os.path.join(output_dir, split, category)

    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "labels.json")

    with open(json_path, "w") as f:
        json.dump(label_dict, f, indent=2)

print("\n✅ All images and labels saved successfully!")