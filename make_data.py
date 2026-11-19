import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)


def generate_shape(label: int, size: int = 16) -> np.ndarray:
    """0 = Circle, 1 = Square, 2 = Triangle"""
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2.0, size / 2.0

    if label == 0:  # Circle
        r = rng.uniform(4.0, 6.0)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        img[mask] = rng.uniform(180, 255)
    elif label == 1:  # Square
        half = rng.uniform(4.0, 6.0)
        mask = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
        img[mask] = rng.uniform(180, 255)
    else:  # Triangle
        h = rng.uniform(9.0, 12.0)
        top_y = 2.0
        mask = (yy >= top_y) & (yy <= top_y + h) & (
            np.abs(xx - cx) <= ((yy - top_y) / h) * (size * 0.45)
        )
        img[mask] = rng.uniform(180, 255)

    # Slight noise
    noise = rng.normal(0, 12, size=(size, size))
    img = np.clip(img + noise, 0, 255)
    return img.flatten()


def make_split(n_per_class: int) -> pd.DataFrame:
    rows = []
    for label in [0, 1, 2]:
        for _ in range(n_per_class):
            pixels = generate_shape(label)
            row = np.append(pixels, label)
            rows.append(row)
    cols = [f"pixel_{i}" for i in range(256)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = df["label"].astype(int)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


train_df = make_split(250)
test_df = make_split(60)

# Cleaning test kosam konni duplicates & NaN values add chesthunnam
train_df = pd.concat([train_df, train_df.iloc[:10]], ignore_index=True)
train_df.loc[0, "pixel_0"] = np.nan
train_df.loc[1, "pixel_5"] = np.nan

train_df.to_csv("data/train_images.csv", index=False)
test_df.to_csv("data/test_images.csv", index=False)
print("Created data/train_images.csv and data/test_images.csv")