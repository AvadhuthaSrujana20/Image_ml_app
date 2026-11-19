import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Output folder
os.makedirs("artifacts", exist_ok=True)

# 1. Train & Test CSVs load cheyyadam
train_df = pd.read_csv("data/train_images.csv")
test_df = pd.read_csv("data/test_images.csv")

print("Before cleaning - Train shape:", train_df.shape, "Test shape:", test_df.shape)

# 2. Data Cleaning (missing values & duplicates remove cheyyadam)
train_df = train_df.dropna().drop_duplicates().reset_index(drop=True)
test_df = test_df.dropna().drop_duplicates().reset_index(drop=True)

print("After cleaning  - Train shape:", train_df.shape, "Test shape:", test_df.shape)

# Cleaned data ni joblib lo save cheyyadam
joblib.dump(train_df, "artifacts/cleaned_train_data.joblib")
joblib.dump(test_df, "artifacts/cleaned_test_data.joblib")

# 3. Features (X) and Target (y) separate cheyyadam
X_full_train = train_df.drop(columns=["label"]).astype(np.float32)
y_full_train = train_df["label"].astype(int)

X_test = test_df.drop(columns=["label"]).astype(np.float32)
y_test = test_df["label"].astype(int)

# 4. Train data nunchi Validation split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train,
    y_full_train,
    test_size=0.2,
    random_state=42,
    stratify=y_full_train,
)

# 5. Preprocessing + ML Model Pipeline
ml_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# 6. Pipeline Run & Validation Accuracy
ml_pipeline.fit(X_train, y_train)
val_preds = ml_pipeline.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# 7. Full cleaned train data tho fit chesi Test Accuracy check
ml_pipeline.fit(X_full_train, y_full_train)
test_preds = ml_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, test_preds))

# 8. Pipeline ni joblib file lo store cheyyadam
MODEL_PATH = "artifacts/image_ml_pipeline.joblib"
joblib.dump(ml_pipeline, MODEL_PATH)
print(f"Model pipeline saved at: {MODEL_PATH}")