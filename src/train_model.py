"""
Week 4: Predictive Modeling and Evaluation
Project: Student Placement Prediction and Skill-Gap Analysis Using Python

Run from the repository root:
    python src/train_model.py
Input:
    data/raw/student_placement_raw_data.csv
Outputs:
    reports/week4_model_metrics.csv
    reports/week4_confusion_matrices.csv
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "student_placement_raw_data.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
before = len(df)
df = df.drop_duplicates().copy()
print(f"Rows before exact-duplicate removal: {before}")
print(f"Rows after exact-duplicate removal: {len(df)}")

target = "placement_status"
features = [
    "department", "cgpa", "backlogs", "internship_experience",
    "projects_completed", "certifications", "aptitude_score",
    "technical_score", "communication_skill"
]
X, y = df[features], df[target]

categorical = [
    "department", "internship_experience", "projects_completed",
    "certifications", "communication_skill"
]
numeric = ["cgpa", "backlogs", "aptitude_score", "technical_score"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    )
}

metric_rows, confusion_rows = [], []
for name, estimator in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", estimator)
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Not Placed)": precision_score(
            y_test, predictions, pos_label="Not Placed", zero_division=0
        ),
        "Recall (Not Placed)": recall_score(
            y_test, predictions, pos_label="Not Placed", zero_division=0
        ),
        "F1-score (Not Placed)": f1_score(
            y_test, predictions, pos_label="Not Placed", zero_division=0
        )
    })
    tp, fn, fp, tn = confusion_matrix(
        y_test, predictions, labels=["Not Placed", "Placed"]
    ).ravel()
    confusion_rows.append({
        "Model": name, "TP (Not Placed)": tp, "FN (Not Placed)": fn,
        "FP (Not Placed)": fp, "TN (Placed)": tn
    })

pd.DataFrame(metric_rows).to_csv(REPORT_DIR / "week4_model_metrics.csv", index=False)
pd.DataFrame(confusion_rows).to_csv(
    REPORT_DIR / "week4_confusion_matrices.csv", index=False
)
print(pd.DataFrame(metric_rows).round(3).to_string(index=False))
print("\nConfusion matrices (Not Placed positive class):")
print(pd.DataFrame(confusion_rows).to_string(index=False))
print("\nNote: metrics are estimates on this dataset's held-out split, not guarantees.")
