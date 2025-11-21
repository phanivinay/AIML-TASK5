# Task5.py - Decision Tree & Random Forest for Classification
# Works even if Graphviz is not installed

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Optional Graphviz
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset loaded successfully!\n")
print(X.head())

# -----------------------------
# 2️⃣ Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Decision Tree Classifier
# -----------------------------
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize Decision Tree (optional)
if GRAPHVIZ_AVAILABLE:
    try:
        dot_data = export_graphviz(
            dt,
            out_file=None,
            feature_names=X.columns,
            class_names=data.target_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree")  # Creates 'decision_tree.pdf'
        print("Decision Tree diagram saved as 'decision_tree.pdf'")
    except Exception as e:
        print("Could not generate Decision Tree diagram:", e)
else:
    print("Graphviz not installed; skipping Decision Tree diagram.")

# -----------------------------
# 4️⃣ Random Forest Classifier
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance
feat_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Feature Importances:\n", feat_importance.head(10))

# -----------------------------
# 5️⃣ Cross-Validation
# -----------------------------
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

