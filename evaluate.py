from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "response_quality.csv").dropna(subset=["prompt", "response", "label"])
df["text"] = "Prompt: " + df["prompt"].astype(str) + " Response: " + df["response"].astype(str)
_, X_test, _, y_test = train_test_split(
    df["text"], df["label"].astype(str), test_size=0.25, random_state=42,
    stratify=df["label"].astype(str)
)
model = joblib.load(ROOT / "models" / "quality_model.joblib")
pred = model.predict(X_test)
print(classification_report(y_test, pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=30)
plt.tight_layout()
out = ROOT / "models" / "confusion_matrix.png"
plt.savefig(out, dpi=160)
print(f"Saved confusion matrix to {out}")
