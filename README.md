# AIML-TASK5
🌳 Task 5: Decision Trees & Random Forests in Python

🚀 Project Overview

This project demonstrates tree-based machine learning models for classification:

Decision Tree Classifier – interpretable model for understanding feature splits.

Random Forest Classifier – ensemble model improving accuracy and robustness.

Cross-validation – validates model performance with k-fold technique.

💡 All datasets used are built-in Python datasets (Breast Cancer Wisconsin), so no external downloads are required.

🛠 Key Features

Train Decision Tree and Random Forest models.

Evaluate using accuracy, precision, recall, F1-score.

Compute cross-validation scores for model reliability.

Display top feature importances from Random Forest.

Optional Decision Tree visualization (requires Graphviz).

Works offline and handles missing Graphviz automatically.

📦 Installation & Setup

Clone the repository:

git clone <your-repo-url>
cd AIML-TASK5


Install dependencies:

pip install pandas scikit-learn graphviz


Optional for visualization:
Download Graphviz: https://graphviz.org/download/

Ensure dot is added to your system PATH.

💻 Usage

Run the main script:

python Task5.py


Expected Output:

Dataset preview (first 5 rows)

Decision Tree accuracy & classification report

Random Forest accuracy & top feature importances

Cross-validation scores

Optional decision_tree.pdf diagram

📊 Visual Output Example
=== Decision Tree Results ===
Accuracy: 0.95

=== Random Forest Results ===
Accuracy: 0.97

Top 10 Feature Importances:
mean concave points      0.15
worst perimeter         0.12
...


Tree diagram saved as decision_tree.pdf if Graphviz is installed.

🎨 Innovative Additions

Automatically skips visualization if Graphviz is missing.

Uses feature importance ranking to highlight influential attributes.

Ready for offline execution without dataset download errors.

Clean, modular Python code for easy customization.

🧠 Insights

Decision Trees provide interpretability but may overfit if not pruned.

Random Forests improve accuracy and generalization through ensembling.

Cross-validation ensures robust performance evaluation.
