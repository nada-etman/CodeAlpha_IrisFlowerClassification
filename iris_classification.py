"""
CodeAlpha - Data Science Internship
Task 1: Iris Flower Classification
------------------------------------
Train a machine learning model to classify Iris flowers (setosa, versicolor,
virginica) based on sepal/petal measurements, then evaluate its performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])  # not a useful feature

print("Dataset shape:", df.shape)
print(df.head())
print(df["Species"].value_counts())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------
sns.pairplot(df, hue="Species")
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns=["Species"]).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("iris_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
X = df.drop(columns=["Species"])
y = df["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------
# 4. Train and compare models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

# ---------------------------------------------------------
# 5. Pick best model and show confusion matrix
# ---------------------------------------------------------
best_name = max(results, key=results.get)
best_model = models[best_name]
best_preds = best_model.predict(X_test)

print(f"\nBest model: {best_name} with accuracy {results[best_name]:.4f}")

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 6. Feature importance (Random Forest)
# ---------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 4))
importances.plot(kind="bar", color="teal")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n=== Model Comparison Summary ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
