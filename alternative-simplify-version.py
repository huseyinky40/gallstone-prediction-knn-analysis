import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore') #todo future update hatalarını kapatmak için yazıldı silinebilir.

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
                             f1_score)

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import shap

# Visualization Settings
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (12, 8)})

# =============================================================================
# PART 0: Load Data
# =============================================================================
try:
    df = pd.read_csv('dataset-uci.csv', sep=';', decimal=',')
    print("Dataset Loaded Successfully!")
except FileNotFoundError:
    print("ERROR: File 'dataset-uci.csv' not found.")
    exit()

# =============================================================================
# PART 1: EDA
# =============================================================================
print("\n--- Starting EDA ---")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gallstone Status', data=df, hue='Gallstone Status', palette='viridis', legend=False)
plt.title('Distribution of Gallstone Status (0: Healthy, 1: Patient)')
plt.show()

# 2. Histograms
numeric_features = ['Age', 'Height', 'Weight', 'Body Mass Index (BMI)']
df[numeric_features].hist(bins=20, figsize=(10, 6), color='#86bf91', zorder=2, rwidth=0.9)
plt.suptitle("Key Demographic Features")
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
print("Generating Correlation Matrix...")
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    k = 15
    cols = numeric_df.corr().nlargest(k, 'Gallstone Status')['Gallstone Status'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=0.9)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8},
                yticklabels=cols.values, xticklabels=cols.values, cmap='coolwarm')
    plt.title('Top 15 Features Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 2: PREPROCESSING
# =============================================================================
X = df.drop('Gallstone Status', axis=1)
y = df['Gallstone Status']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # %20 test - %80 eğitim verisi

# =============================================================================
# PART 3: TUNING & CV
# =============================================================================
print("\n--- Training Models (Hyperparameter Tuning & CV) ---")

base_models = { #todo ekstra model eklenebilr
    "k-NN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Multilayer Perceptron": MLPClassifier(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42)
}

param_grids = {
    "k-NN": {'n_neighbors': range(3, 20, 2)},
    "Random Forest": {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
    "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    "Support Vector Machine": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Multilayer Perceptron": {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.01]},
    "Decision Tree": {'max_depth': [None, 10, 20]},
    "AdaBoost": {'n_estimators': [50, 100]},
    "Logistic Regression": {'C': [0.1, 1, 10]}
}

results = []
trained_models = {}
confusion_matrices = {}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    trained_models[name] = best_model

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = [0] * len(y_test)
        roc = 0

    results.append({
        "Model": name,
        "CV Mean Accuracy": cv_scores.mean(),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc
    })

    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# --- Ensemble ---
print("Training Ensemble Model...")
voting_clf = VotingClassifier( #todo 3 modele düşürülebilir
    estimators=[
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('knn', trained_models['k-NN']),
        ('svm', trained_models['Support Vector Machine'])
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

y_pred_ens = voting_clf.predict(X_test)
y_prob_ens = voting_clf.predict_proba(X_test)[:, 1]
cv_ens = cross_val_score(voting_clf, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()

confusion_matrices["Ensemble"] = confusion_matrix(y_test, y_pred_ens)
trained_models["Ensemble"] = voting_clf

results.append({
    "Model": "Ensemble (Voting)",
    "CV Mean Accuracy": cv_ens,
    "Test Accuracy": accuracy_score(y_test, y_pred_ens),
    "F1 Score": f1_score(y_test, y_pred_ens),
    "ROC-AUC": roc_auc_score(y_test, y_prob_ens)
})

# =============================================================================
# PART 4: TABLES
# =============================================================================

# 1. Model Performance Table
results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
print("\n" + "=" * 90)
print("                       SCIENTIFIC MODEL PERFORMANCE REPORT                       ")
print("=" * 90)
print(results_df.to_string(index=False, formatters={
    'CV Mean Accuracy': '{:.4f}'.format,
    'Test Accuracy': '{:.4f}'.format,
    'F1 Score': '{:.4f}'.format,
    'ROC-AUC': '{:.4f}'.format
}))
print("=" * 90 + "\n")

# 2. Feature Importance Table
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20) #ascending=True

print("\n" + "=" * 50)
print("     TOP 20 FEATURE IMPORTANCE (RANDOM FOREST)     ")
print("=" * 50)
print(feature_imp_df.to_string(index=False, formatters={'Importance': '{:.6f}'.format}))
print("=" * 50 + "\n")

# =============================================================================
# PART 5: VISUALIZATIONS
# =============================================================================

# 1. Confusion Matrices
print("Generating Confusion Matrices...")
num_models = len(confusion_matrices)
cols = 3 #todo 4 sütun dene
rows = math.ceil(num_models / cols)
plt.figure(figsize=(15, 4 * rows))
for i, (name, cm) in enumerate(confusion_matrices.items()):
    plt.subplot(rows, cols, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 2. Performance Bar Plot
plt.figure(figsize=(12, 6))
melted_df = results_df.melt(id_vars="Model", value_vars=["CV Mean Accuracy", "Test Accuracy"], var_name="Metric",
                            value_name="Score")
sns.barplot(x='Score', y='Model', hue='Metric', data=melted_df, palette='viridis')
plt.title('Cross-Validation vs Test Accuracy')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()

# 3. ROC Curve
plt.figure(figsize=(10, 8)) # figsize=(12,8)
top_models_list = results_df['Model'].head(4).tolist()
for name in top_models_list:
    if name in trained_models:
        model = trained_models[name]
        if hasattr(model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            plt.plot(fpr, tpr, lw=2, label=f'{name}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve Analysis (Top Models)')
plt.legend(loc="lower right")
plt.show()

# 4. Feature Importance Bar Plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 20 Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# =============================================================================
# PART 6: SHAP
# =============================================================================
print("\nPerforming SHAP Analysis...")
try:
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
        expected_val_class1 = explainer.expected_value[1]

    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            shap_vals_class1 = shap_values[:, :, 1]

            if isinstance(explainer.expected_value, np.ndarray) or isinstance(explainer.expected_value, list):
                expected_val_class1 = explainer.expected_value[1]
            else:
                expected_val_class1 = explainer.expected_value
        else:
            shap_vals_class1 = shap_values
            expected_val_class1 = explainer.expected_value
    else:
        shap_vals_class1 = shap_values
        expected_val_class1 = explainer.expected_value

    plt.figure()
    shap.summary_plot(shap_vals_class1, X_test, show=False)
    plt.title("SHAP Global Feature Impact (Beeswarm)")
    plt.tight_layout()
    plt.show()

    positive_indices = np.where(y_test == 1)[0]

    if len(positive_indices) > 0:
        patient_idx = positive_indices[0]  # İlk hasta örneği
        print(f"Generating Waterfall Plot for Patient Index: {patient_idx}")

        plt.figure()
        shap_explanation = shap.Explanation(values=shap_vals_class1[patient_idx],
                                            base_values=expected_val_class1,
                                            data=X_test.iloc[patient_idx],
                                            feature_names=X.columns)

        shap.plots.waterfall(shap_explanation, show=False)
        plt.title(f"Local Explanation for Patient #{patient_idx}", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("waterfall plot not generated")

except Exception as e:
    print(f"Error: {e}")