# src/evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib
import os

# Load the test data and model
print("="*50)
print("PHASE 6: DEEPER MODEL EVALUATION")
print("="*50)

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()  # Convert to Series

# Load the best model
best_model = joblib.load('models/best_model.pkl')
print(f"Best model loaded: {type(best_model).__name__}")

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 1. CONFUSION MATRIX
print("\n1. CONFUSION MATRIX")
print("-" * 40)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives (Correctly predicted No Disease): {tn}")
print(f"False Positives (Incorrectly predicted Disease): {fp}")
print(f"False Negatives (Incorrectly predicted No Disease): {fn} ← MOST IMPORTANT!")
print(f"True Positives (Correctly predicted Disease): {tp}")

# Calculate metrics from confusion matrix
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\nMetrics from Confusion Matrix:")
print(f"Recall (Sensitivity): {recall:.4f} ← Our primary metric!")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {type(best_model).__name__}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=100)
plt.show()
print("\nConfusion matrix saved to models/confusion_matrix.png")

# 2. ROC CURVE
print("\n2. ROC CURVE")
print("-" * 40)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title(f'ROC Curve - {type(best_model).__name__}')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=100)
plt.show()
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("ROC curve saved to models/roc_curve.png")

# 3. FEATURE IMPORTANCE (if available)
print("\n3. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

if hasattr(best_model, 'feature_importances_'):
    # For tree-based models
    feature_names = X_test.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances - {type(best_model).__name__}')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=100)
    plt.show()
    
    print("Top 5 most important features:")
    for i in range(min(5, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
elif hasattr(best_model, 'coef_'):
    # For linear models
    feature_names = X_test.columns
    coefficients = best_model.coef_[0]
    indices = np.argsort(np.abs(coefficients))[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in coefficients[indices]]
    plt.bar(range(len(coefficients)), coefficients[indices], color=colors)
    plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Coefficients - {type(best_model).__name__} (Green=Positive, Red=Negative)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('models/coefficients.png', dpi=100)
    plt.show()
    
    print("Top 5 features by coefficient magnitude:")
    for i in range(min(5, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {coefficients[indices[i]]:.4f}")
else:
    print("Feature importance not available for this model type.")

# 4. THRESHOLD OPTIMIZATION (for Recall)
print("\n4. THRESHOLD OPTIMIZATION")
print("-" * 40)
print("We can adjust the decision threshold to improve Recall")
print("Current threshold: 0.5")

# Try different thresholds
thresholds_to_try = np.arange(0.3, 0.7, 0.05)
threshold_results = []

for threshold in thresholds_to_try:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    tn_th, fp_th, fn_th, tp_th = confusion_matrix(y_test, y_pred_threshold).ravel()
    recall_th = tp_th / (tp_th + fn_th)
    precision_th = tp_th / (tp_th + fp_th)
    threshold_results.append({
        'Threshold': threshold,
        'Recall': recall_th,
        'Precision': precision_th,
        'False Negatives': fn_th
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nEffect of different thresholds:")
print(threshold_df.round(4))

# Visualize threshold trade-off
plt.figure(figsize=(10, 6))
plt.plot(threshold_df['Threshold'], threshold_df['Recall'], 'b-o', label='Recall')
plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 'r-o', label='Precision')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Recall-Precision Trade-off at Different Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/threshold_analysis.png', dpi=100)
plt.show()

print("\n✅ PHASE 6 COMPLETE!")
print("="*50)