"""
╔══════════════════════════════════════════════════════════════════════╗
║         DECODELABS - DATA SCIENCE PROJECT 2                          ║
║         Supervised Learning: Fraud Detection Pipeline                ║
║         Dataset: E-Commerce Orders (1200 records)                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("✅ All libraries imported successfully.\n")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: LOAD & EXPLORE DATA
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 2: LOADING AND EXPLORING DATA")
print("=" * 60)

df = pd.read_excel('/mnt/user-data/uploads/Dataset_for_Data_Analytics__1_.xlsx')

print(f"\n📦 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🔎 Columns:\n{df.dtypes.to_string()}")
print(f"\n📊 Missing Values:\n{df.isnull().sum().to_string()}")
print(f"\n📈 OrderStatus Distribution:\n{df['OrderStatus'].value_counts().to_string()}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: ENGINEER FRAUD LABEL
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
Real fraud datasets have a binary target column (0 = legit, 1 = fraud).
Our e-commerce dataset doesn't have one, so we CREATE a realistic fraud
label by combining multiple suspicious signals:

  - Cancelled or Returned orders with high TotalPrice (suspicious refund)
  - Payment via Gift Card (commonly used in fraud)
  - Very high quantity (bulk-order fraud pattern)
  - High UnitPrice + high quantity (high-value fraud)

This gives us a ~5-8% fraud rate — realistic for retail fraud datasets.
"""
print("\n" + "=" * 60)
print("SECTION 3: ENGINEERING FRAUD LABEL")
print("=" * 60)

# Flag suspicious combinations
fraud_mask = (
    (df['OrderStatus'].isin(['Cancelled', 'Returned']) & (df['TotalPrice'] > 2000)) |
    ((df['PaymentMethod'] == 'Gift Card') & (df['TotalPrice'] > 1500)) |
    ((df['Quantity'] >= 5) & (df['UnitPrice'] > 500)) |
    ((df['PaymentMethod'] == 'Gift Card') & (df['OrderStatus'] == 'Cancelled'))
)

df['IsFraud'] = fraud_mask.astype(int)

fraud_counts = df['IsFraud'].value_counts()
fraud_pct = df['IsFraud'].mean() * 100

print(f"\n🚨 Fraud Label Distribution:")
print(f"   Legitimate (0): {fraud_counts[0]} ({100 - fraud_pct:.2f}%)")
print(f"   Fraudulent  (1): {fraud_counts[1]} ({fraud_pct:.2f}%)")
print(f"\n   → This is a highly imbalanced dataset (like the real world!)")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
We drop columns that are identifiers (not predictive) and encode
categorical columns into numbers so ML models can use them.
"""
print("\n" + "=" * 60)
print("SECTION 4: FEATURE ENGINEERING")
print("=" * 60)

# Drop non-predictive identifier columns
drop_cols = ['OrderID', 'CustomerID', 'ShippingAddress', 'TrackingNumber', 'Date']
df_model = df.drop(columns=drop_cols)

# Fill missing CouponCode (309 nulls) with 'None'
df_model['CouponCode'] = df_model['CouponCode'].fillna('None')

# Label-encode all remaining categorical columns
cat_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'CouponCode', 'ReferralSource']
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

print(f"\n✅ Processed features:\n{df_model.drop('IsFraud', axis=1).dtypes.to_string()}")

# Define features and target
X = df_model.drop('IsFraud', axis=1)
y = df_model['IsFraud']

print(f"\n📐 Feature matrix X: {X.shape}")
print(f"🎯 Target vector y: {y.shape} | Fraud rate: {y.mean()*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: STRATIFIED TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
We split BEFORE applying SMOTE. This is the Golden Rule:
  - SMOTE only touches training data
  - Test data stays 100% real, imbalanced (reflecting real world)
  - stratify=y ensures both splits have the same fraud ratio
"""
print("\n" + "=" * 60)
print("SECTION 5: STRATIFIED TRAIN/TEST SPLIT (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # <- Preserves fraud ratio in both splits
)

print(f"\n📊 Training Set: {X_train.shape[0]} samples | Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"📊 Test Set:     {X_test.shape[0]} samples  | Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
print("\n✅ SMOTE will ONLY be applied inside the pipeline on training data.")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: BUILD PIPELINES (LEAK-FREE)
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
We use imblearn.pipeline.Pipeline (NOT sklearn's Pipeline).
This is critical — it understands SMOTE's fit_resample() interface.

Pipeline 1 (Logistic Regression):
  StandardScaler → SMOTE → LogisticRegression
  (LR needs scaling; features like UnitPrice can be 0-700, distorting regularization)

Pipeline 2 (Random Forest):
  SMOTE → RandomForestClassifier
  (RF is scale-invariant; no scaler needed)
"""
print("\n" + "=" * 60)
print("SECTION 6: BUILDING IMBLEARN PIPELINES")
print("=" * 60)

# --- Pipeline 1: Logistic Regression ---
lr_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),           # Step 1: Normalize features
    ('smote', SMOTE(random_state=42)),      # Step 2: Oversample minority class
    ('classifier', LogisticRegression(     # Step 3: Train model
        max_iter=1000,
        random_state=42
    ))
])

# --- Pipeline 2: Random Forest ---
rf_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),      # Step 1: Oversample minority class
    ('classifier', RandomForestClassifier( # Step 2: Train ensemble model
        random_state=42
    ))
])

print("\n✅ Logistic Regression Pipeline: [StandardScaler → SMOTE → LogisticRegression]")
print("✅ Random Forest Pipeline:        [SMOTE → RandomForestClassifier]")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: HYPERPARAMETER TUNING WITH GridSearchCV
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
GridSearchCV tests every combination of hyperparameters using 5-fold
cross-validation. Because SMOTE is INSIDE the pipeline, it is applied
only on each training fold — never on the validation fold.
This is zero-leakage tuning.

We score on 'recall' because in fraud detection, missing a fraud
(False Negative) is far more costly than a false alarm.
"""
print("\n" + "=" * 60)
print("SECTION 7: HYPERPARAMETER TUNING (GridSearchCV, 5-Fold CV)")
print("=" * 60)

# --- GridSearch for Logistic Regression ---
lr_param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__C': [0.01, 0.1, 1.0]
}

print("\n🔍 Tuning Logistic Regression (9 combinations × 5 folds = 45 fits)...")
lr_grid = GridSearchCV(
    lr_pipeline,
    param_grid=lr_param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print(f"   ✅ Best Params: {lr_grid.best_params_}")
print(f"   ✅ Best CV Recall: {lr_grid.best_score_:.4f}")

# --- GridSearch for Random Forest ---
rf_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

print("\n🔍 Tuning Random Forest (12 combinations × 5 folds = 60 fits)...")
rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"   ✅ Best Params: {rf_grid.best_params_}")
print(f"   ✅ Best CV Recall: {rf_grid.best_score_:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: EVALUATE ON HELD-OUT TEST SET
# ═══════════════════════════════════════════════════════════════════════
"""
EXPLANATION:
We evaluate on the ORIGINAL imbalanced test set (real-world conditions).
We report Precision, Recall, F1, and ROC-AUC — NOT accuracy.

Key definitions:
  Precision = TP / (TP + FP) → When we flag fraud, how often are we right?
  Recall    = TP / (TP + FN) → Of all actual fraud, how many did we catch?
  F1        = Harmonic mean of Precision and Recall
  ROC-AUC   = Overall ability to separate fraud from legit (target: > 0.85)
"""
print("\n" + "=" * 60)
print("SECTION 8: FINAL EVALUATION ON TEST SET")
print("=" * 60)

best_lr = lr_grid.best_estimator_
best_rf = rf_grid.best_estimator_

y_pred_lr = best_lr.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

y_prob_lr = best_lr.predict_proba(X_test)[:, 1]
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

def print_metrics(name, y_true, y_pred, y_prob):
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))

print_metrics("LOGISTIC REGRESSION", y_test, y_pred_lr, y_prob_lr)
print_metrics("RANDOM FOREST",       y_test, y_pred_rf, y_prob_rf)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 9: GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("DecodeLabs Project 2 — Fraud Detection Pipeline Results",
             fontsize=16, fontweight='bold', y=1.01)

# ── Plot 1: Class Imbalance ──────────────────────────────────────────
ax1 = axes[0, 0]
labels = ['Legitimate', 'Fraud']
counts = [fraud_counts[0], fraud_counts[1]]
colors = ['#4CAF50', '#F44336']
bars = ax1.bar(labels, counts, color=colors, edgecolor='black', width=0.5)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{count}\n({count/sum(counts)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')
ax1.set_title("Class Distribution (Imbalanced)", fontweight='bold')
ax1.set_ylabel("Count")
ax1.set_ylim(0, max(counts) * 1.2)

# ── Plot 2: Confusion Matrix — Logistic Regression ──────────────────
ax2 = axes[0, 1]
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
ax2.set_title("Confusion Matrix\nLogistic Regression", fontweight='bold')
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")

# ── Plot 3: Confusion Matrix — Random Forest ────────────────────────
ax3 = axes[0, 2]
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
ax3.set_title("Confusion Matrix\nRandom Forest", fontweight='bold')
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

# ── Plot 4: ROC Curve Comparison ────────────────────────────────────
ax4 = axes[1, 0]
RocCurveDisplay.from_predictions(y_test, y_prob_lr, name="Logistic Regression",
                                  ax=ax4, color='steelblue')
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest",
                                  ax=ax4, color='darkorange')
ax4.plot([0,1],[0,1],'k--', label='Random (AUC=0.5)')
ax4.set_title("ROC Curve Comparison", fontweight='bold')
ax4.legend(fontsize=9)

# ── Plot 5: Metric Comparison Bar Chart ─────────────────────────────
ax5 = axes[1, 1]
metrics_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = [
    precision_score(y_test, y_pred_lr),
    recall_score(y_test, y_pred_lr),
    f1_score(y_test, y_pred_lr),
    roc_auc_score(y_test, y_prob_lr)
]
rf_scores = [
    precision_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_rf),
    f1_score(y_test, y_pred_rf),
    roc_auc_score(y_test, y_prob_rf)
]
x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax5.bar(x - width/2, lr_scores, width, label='Logistic Regression',
                color='steelblue', edgecolor='black')
bars2 = ax5.bar(x + width/2, rf_scores, width, label='Random Forest',
                color='darkorange', edgecolor='black')
for bar in list(bars1) + list(bars2):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names)
ax5.set_ylim(0, 1.15)
ax5.set_title("Model Performance Comparison\n(No Accuracy — Only Real Metrics)", fontweight='bold')
ax5.legend()
ax5.set_ylabel("Score")

# ── Plot 6: Feature Importance (RF) ─────────────────────────────────
ax6 = axes[1, 2]
feature_names = X.columns.tolist()
importances = best_rf.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]
ax6.barh([feature_names[i] for i in indices],
         [importances[i] for i in indices],
         color='darkorange', edgecolor='black')
ax6.set_title("Feature Importances\n(Random Forest)", fontweight='bold')
ax6.set_xlabel("Importance Score")
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fraud_detection_results.png',
            dpi=150, bbox_inches='tight')
print("✅ Visualization saved: fraud_detection_results.png")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 10: FINAL SUMMARY")
print("=" * 60)

lr_auc = roc_auc_score(y_test, y_prob_lr)
rf_auc = roc_auc_score(y_test, y_prob_rf)
winner = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │               PIPELINE SUMMARY                      │
  ├─────────────────────────────────────────────────────┤
  │ Dataset        : 1200 e-commerce orders              │
  │ Fraud Rate     : {fraud_pct:.2f}% (imbalanced)              │
  │ Resampling     : SMOTE (inside pipeline, no leakage) │
  │ Validation     : 5-Fold GridSearchCV on recall       │
  │ Test Set       : Original imbalanced distribution    │
  ├─────────────────────────────────────────────────────┤
  │  Model               Recall    ROC-AUC               │
  │  Logistic Regression  {recall_score(y_test,y_pred_lr):.4f}    {lr_auc:.4f}         │
  │  Random Forest        {recall_score(y_test,y_pred_rf):.4f}    {rf_auc:.4f}         │
  ├─────────────────────────────────────────────────────┤
  │  🏆 Winner: {winner:<42}│
  └─────────────────────────────────────────────────────┘
  
  ✅ Accuracy metric was NEVER used.
  ✅ SMOTE applied AFTER train/test split (zero data leakage).
  ✅ imblearn.pipeline.Pipeline used (not sklearn's).
""")
