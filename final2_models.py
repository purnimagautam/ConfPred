# Extended version of your code with multiple models, hyperparameter tuning, combined plots, and file output for publication

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Output directory
os.makedirs("model_outputs", exist_ok=True)

# Load and prepare data
df = pd.read_csv("total_merged.csv")
df = df.iloc[1:, :]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# PCA
# pca = PCA(n_components=100)
# X_train_pca = pca.fit_transform(X_train_smote)
# X_test_pca = pca.transform(X_test)

rfe_estimator = LogisticRegression(solver='liblinear', max_iter=1000)
selector = RFE(estimator=rfe_estimator, n_features_to_select=100)
X_train_pca = selector.fit_transform(X_train_smote, y_train_smote)
X_test_pca = selector.transform(X_test)


# Define models with hyperparameter grids
models = {
    "Logistic Regression": (LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
                             {'C': [0.1, 1, 10]}),

    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {'max_depth': [5, 10, 15, 20, 30], 'min_samples_split': [2, 10]}),

    "Random Forest": (RandomForestClassifier(random_state=42),
                      {'n_estimators': [100, 150, 200, 250, 300], 'max_depth': [5, 10, 15, 20, 30]}),

    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'n_estimators': [100, 150, 200, 250, 300]}),

    "SVM": (SVC(kernel='rbf', probability=True, random_state=42),
            {'C': [0.1, 1, 10]}),

    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                {'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]})
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

# For plotting
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10.colors

for i, (name, (model, param_grid)) in enumerate(models.items()):
    print(f"\n==== {name} ====")
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=cv, n_jobs=-1)
    grid.fit(X_train_pca, y_train_smote)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_pca)
    y_proba = best_model.predict_proba(X_test_pca)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    result = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "MCC": mcc,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Best Params": grid.best_params_
    }
    all_results.append(result)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", color=colors[i % len(colors)])

    # Save PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend()
    plt.savefig(f"model_outputs/PR_{name.replace(' ', '_')}.png")
    plt.close()

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train_pca, y_train_smote, cv=cv, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.savefig(f"model_outputs/LearningCurve_{name.replace(' ', '_')}.png")
    plt.close()

# Final ROC plot
# plt.plot([0, 1], [0, 1], 'k--', label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves - All Models")
# plt.legend()
# plt.tight_layout()
# plt.savefig("model_outputs/ROC_Curves_All.png")
# plt.show()

plt.figure(figsize=(6, 5))  # good size for 2-column paper
# Plot ROC for random classifier
plt.plot([0, 1], [0, 1], 'k--', label="Random")
# Set axis labels with larger font
plt.xlabel("False Positive Rate", fontsize=10)
plt.ylabel("True Positive Rate", fontsize=10)
# Set title with larger font
plt.title("ROC Curves - All Models", fontsize=10)
# Customize tick font sizes
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
# Adjust legend font size
plt.legend(fontsize=9, loc="lower right")
# Improve layout for better spacing
plt.tight_layout()
# Save at high DPI for clarity in print
plt.savefig("model_outputs/ROC_Curves_All.png", dpi=300)
# Show the plot
plt.show()

# Save results
results_df = pd.DataFrame(all_results)
results_df.to_csv("model_outputs/All_Model_Results.csv", index=False)
print("\nAll model results saved to 'model_outputs/All_Model_Results.csv'")
