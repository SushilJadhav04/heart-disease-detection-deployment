# src/optimize_threshold.py (IMPROVED VERSION)
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import os
import matplotlib.pyplot as plt

print("="*60)
print("OPTIMIZING THRESHOLD FOR DEPLOYMENT")
print("="*60)

# ============================================
# TRY LOADING ENHANCED MODEL FIRST, THEN FALLBACK
# ============================================
print("\n📂 Loading model and data...")

# Try enhanced model first
enhanced_path = 'models/enhanced/best_model.pkl'
original_path = 'models/best_model.pkl'
enhanced_data_path = 'data/processed/enhanced/X_test.csv'
original_data_path = 'data/processed/X_test.csv'

model = None
X_test = None
y_test = None
is_enhanced = False

# Try enhanced model
if os.path.exists(enhanced_path) and os.path.exists(enhanced_data_path):
    try:
        model = joblib.load(enhanced_path)
        X_test = pd.read_csv(enhanced_data_path)
        y_test = pd.read_csv('data/processed/enhanced/y_test.csv').squeeze()
        is_enhanced = True
        print("✅ Loaded ENHANCED model and data")
    except Exception as e:
        print(f"⚠️ Could not load enhanced model: {e}")

# Fallback to original
if model is None:
    try:
        model = joblib.load(original_path)
        X_test = pd.read_csv(original_data_path)
        y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
        print("✅ Loaded ORIGINAL model and data")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run model_training.py first")
        exit(1)

print(f"Model type: {type(model).__name__}")
print(f"Test data shape: {X_test.shape}")

# ============================================
# GET PREDICTION PROBABILITIES
# ============================================
print("\n📊 Getting prediction probabilities...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"Mean probability: {y_pred_proba.mean():.4f}")
print(f"Median probability: {np.median(y_pred_proba):.4f}")

# ============================================
# TEST MULTIPLE THRESHOLDS
# ============================================
print("\n🔍 Testing multiple thresholds...")

# Test thresholds from 0.20 to 0.65 with smaller steps
thresholds = np.arange(0.20, 0.65, 0.01)
results = []

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    results.append({
        'Threshold': threshold,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'False Negatives': fn,
        'False Positives': fp,
        'True Positives': tp,
        'True Negatives': tn
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# ============================================
# FIND OPTIMAL THRESHOLDS FOR DIFFERENT GOALS
# ============================================
print("\n" + "="*60)
print("OPTIMAL THRESHOLDS FOR DIFFERENT GOALS")
print("="*60)

# 1. Best F1-Score (Balance between Recall and Precision)
best_f1_idx = results_df['F1-Score'].idxmax()
threshold_f1 = results_df.loc[best_f1_idx, 'Threshold']
recall_f1 = results_df.loc[best_f1_idx, 'Recall']
precision_f1 = results_df.loc[best_f1_idx, 'Precision']
f1_f1 = results_df.loc[best_f1_idx, 'F1-Score']
fn_f1 = results_df.loc[best_f1_idx, 'False Negatives']
fp_f1 = results_df.loc[best_f1_idx, 'False Positives']

print(f"\n1️⃣  OPTIMAL FOR BALANCE (Best F1-Score):")
print(f"   Threshold:      {threshold_f1:.2f}")
print(f"   Recall:         {recall_f1:.4f} ({recall_f1*100:.1f}%)")
print(f"   Precision:      {precision_f1:.4f} ({precision_f1*100:.1f}%)")
print(f"   F1-Score:       {f1_f1:.4f}")
print(f"   False Negatives: {int(fn_f1)}")
print(f"   False Positives: {int(fp_f1)}")

# 2. High Recall (Catch most disease cases)
high_recall_thresholds = results_df[results_df['Recall'] >= 0.95]
if not high_recall_thresholds.empty:
    best_high_recall_idx = high_recall_thresholds['F1-Score'].idxmax()
    threshold_hr = results_df.loc[best_high_recall_idx, 'Threshold']
    recall_hr = results_df.loc[best_high_recall_idx, 'Recall']
    precision_hr = results_df.loc[best_high_recall_idx, 'Precision']
    f1_hr = results_df.loc[best_high_recall_idx, 'F1-Score']
    fn_hr = results_df.loc[best_high_recall_idx, 'False Negatives']
    fp_hr = results_df.loc[best_high_recall_idx, 'False Positives']
    
    print(f"\n2️⃣  OPTIMAL FOR HIGH RECALL (≥95%):")
    print(f"   Threshold:      {threshold_hr:.2f}")
    print(f"   Recall:         {recall_hr:.4f} ({recall_hr*100:.1f}%)")
    print(f"   Precision:      {precision_hr:.4f} ({precision_hr*100:.1f}%)")
    print(f"   F1-Score:       {f1_hr:.4f}")
    print(f"   False Negatives: {int(fn_hr)}")
    print(f"   False Positives: {int(fp_hr)}")

# 3. High Precision (Fewer false alarms)
high_precision_thresholds = results_df[results_df['Precision'] >= 0.70]
if not high_precision_thresholds.empty:
    best_high_precision_idx = high_precision_thresholds['F1-Score'].idxmax()
    threshold_hp = results_df.loc[best_high_precision_idx, 'Threshold']
    recall_hp = results_df.loc[best_high_precision_idx, 'Recall']
    precision_hp = results_df.loc[best_high_precision_idx, 'Precision']
    f1_hp = results_df.loc[best_high_precision_idx, 'F1-Score']
    fn_hp = results_df.loc[best_high_precision_idx, 'False Negatives']
    fp_hp = results_df.loc[best_high_precision_idx, 'False Positives']
    
    print(f"\n3️⃣  OPTIMAL FOR HIGH PRECISION (≥70%):")
    print(f"   Threshold:      {threshold_hp:.2f}")
    print(f"   Recall:         {recall_hp:.4f} ({recall_hp*100:.1f}%)")
    print(f"   Precision:      {precision_hp:.4f} ({precision_hp*100:.1f}%)")
    print(f"   F1-Score:       {f1_hp:.4f}")
    print(f"   False Negatives: {int(fn_hp)}")
    print(f"   False Positives: {int(fp_hp)}")

# ============================================
# RECOMMENDED THRESHOLD
# ============================================
print("\n" + "="*60)
print("🎯 RECOMMENDED THRESHOLD")
print("="*60)

# For healthcare, we want high recall with decent precision
# Choose threshold that gives recall >= 0.90 with best F1
recommended_df = results_df[results_df['Recall'] >= 0.90]
if not recommended_df.empty:
    best_recommended_idx = recommended_df['F1-Score'].idxmax()
    RECOMMENDED_THRESHOLD = results_df.loc[best_recommended_idx, 'Threshold']
    recommended_recall = results_df.loc[best_recommended_idx, 'Recall']
    recommended_precision = results_df.loc[best_recommended_idx, 'Precision']
    recommended_f1 = results_df.loc[best_recommended_idx, 'F1-Score']
    recommended_fn = results_df.loc[best_recommended_idx, 'False Negatives']
    recommended_fp = results_df.loc[best_recommended_idx, 'False Positives']
else:
    # Fallback to best F1
    RECOMMENDED_THRESHOLD = threshold_f1
    recommended_recall = recall_f1
    recommended_precision = precision_f1
    recommended_f1 = f1_f1
    recommended_fn = fn_f1
    recommended_fp = fp_f1

print(f"\n✅ RECOMMENDED THRESHOLD: {RECOMMENDED_THRESHOLD:.2f}")
print(f"   Based on: Recall ≥ 90% with best F1-Score")
print(f"\n   Performance at this threshold:")
print(f"   • Recall:         {recommended_recall:.4f} ({recommended_recall*100:.1f}%)")
print(f"   • Precision:      {recommended_precision:.4f} ({recommended_precision*100:.1f}%)")
print(f"   • F1-Score:       {recommended_f1:.4f}")
print(f"   • False Negatives: {int(recommended_fn)} (missed patients)")
print(f"   • False Positives: {int(recommended_fp)} (false alarms)")

# ============================================
# COMPARE WITH DEFAULT THRESHOLD (0.50)
# ============================================
print("\n" + "="*60)
print("📊 COMPARISON WITH DEFAULT THRESHOLD (0.50)")
print("="*60)

# Get metrics at 0.50
threshold_50_results = results_df[results_df['Threshold'] == 0.50]
if not threshold_50_results.empty:
    recall_50 = threshold_50_results['Recall'].values[0]
    precision_50 = threshold_50_results['Precision'].values[0]
    f1_50 = threshold_50_results['F1-Score'].values[0]
    fn_50 = threshold_50_results['False Negatives'].values[0]
    fp_50 = threshold_50_results['False Positives'].values[0]
    
    print(f"\nAt threshold 0.50:")
    print(f"   Recall:         {recall_50:.4f} ({recall_50*100:.1f}%)")
    print(f"   Precision:      {precision_50:.4f} ({precision_50*100:.1f}%)")
    print(f"   False Negatives: {int(fn_50)}")
    print(f"   False Positives: {int(fp_50)}")
    
    print(f"\nAt recommended threshold {RECOMMENDED_THRESHOLD:.2f}:")
    print(f"   Recall:         {recommended_recall:.4f} ({recommended_recall*100:.1f}%)")
    print(f"   Precision:      {recommended_precision:.4f} ({recommended_precision*100:.1f}%)")
    print(f"   False Negatives: {int(recommended_fn)}")
    print(f"   False Positives: {int(recommended_fp)}")
    
    print(f"\nImprovement with optimized threshold:")
    print(f"   Recall change:  {recommended_recall - recall_50:+.4f} ({(recommended_recall - recall_50)*100:+.1f}%)")
    print(f"   FN change:      {int(recommended_fn - fn_50):+d} patients")

# ============================================
# SAVE THRESHOLD INFO
# ============================================
print("\n" + "="*60)
print("💾 SAVING THRESHOLD INFORMATION")
print("="*60)

# Determine save path based on model type
if is_enhanced:
    save_dir = 'models/enhanced'
else:
    save_dir = 'models'

os.makedirs(save_dir, exist_ok=True)

# Save recommended threshold
threshold_info = {
    'optimal_threshold': float(RECOMMENDED_THRESHOLD),
    'recall_at_optimal': float(recommended_recall),
    'precision_at_optimal': float(recommended_precision),
    'f1_at_optimal': float(recommended_f1),
    'false_negatives': int(recommended_fn),
    'false_positives': int(recommended_fp),
    'test_size': len(y_test),
    'model_type': type(model).__name__,
    'is_enhanced': is_enhanced
}

joblib.dump(threshold_info, f'{save_dir}/threshold.pkl')
print(f"✅ Threshold info saved to {save_dir}/threshold.pkl")

# Save full analysis
results_df.to_csv(f'{save_dir}/threshold_analysis.csv', index=False)
print(f"✅ Full threshold analysis saved to {save_dir}/threshold_analysis.csv")

# ============================================
# VISUALIZE THRESHOLD ANALYSIS
# ============================================
print("\n📈 Generating visualization...")

try:
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Recall-Precision-F1 vs Threshold
    plt.subplot(2, 2, 1)
    plt.plot(results_df['Threshold'], results_df['Recall'], 'b-', linewidth=2, label='Recall')
    plt.plot(results_df['Threshold'], results_df['Precision'], 'r-', linewidth=2, label='Precision')
    plt.plot(results_df['Threshold'], results_df['F1-Score'], 'g-', linewidth=2, label='F1-Score')
    plt.axvline(x=RECOMMENDED_THRESHOLD, color='black', linestyle='--', label=f'Optimal ({RECOMMENDED_THRESHOLD:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Recall-Precision Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: False Negatives vs False Positives
    plt.subplot(2, 2, 2)
    plt.plot(results_df['Threshold'], results_df['False Negatives'], 'b-', linewidth=2, label='False Negatives (Missed)')
    plt.plot(results_df['Threshold'], results_df['False Positives'], 'r-', linewidth=2, label='False Positives (False Alarms)')
    plt.axvline(x=RECOMMENDED_THRESHOLD, color='black', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Errors vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix at Optimal Threshold
    plt.subplot(2, 2, 3)
    cm = np.array([[int(recommended_fn), int(recommended_fp)], 
                   [int(recommended_fn), int(recommended_fp)]])  # Placeholder
    # Get actual confusion matrix
    y_pred_optimal = (y_pred_proba >= RECOMMENDED_THRESHOLD).astype(int)
    cm_actual = confusion_matrix(y_test, y_pred_optimal)
    
    import seaborn as sns
    sns.heatmap(cm_actual, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title(f'Confusion Matrix at Threshold {RECOMMENDED_THRESHOLD:.2f}')
    
    # Plot 4: Threshold Performance Summary
    plt.subplot(2, 2, 4)
    metrics = ['Recall', 'Precision', 'F1-Score']
    values = [recommended_recall, recommended_precision, recommended_f1]
    colors = ['blue', 'red', 'green']
    plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Performance at Optimal Threshold')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/threshold_optimization.png', dpi=100, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_dir}/threshold_optimization.png")
    
    # Show plot
    plt.show()
except Exception as e:
    print(f"⚠️ Could not generate visualization: {e}")

# ============================================
# PRINT SUMMARY TABLE
# ============================================
print("\n" + "="*60)
print("📋 TOP 5 THRESHOLDS BY F1-SCORE")
print("="*60)
print(results_df.nlargest(5, 'F1-Score')[['Threshold', 'Recall', 'Precision', 'F1-Score', 'False Negatives', 'False Positives']].round(4).to_string())

print("\n" + "="*60)
print("✅ THRESHOLD OPTIMIZATION COMPLETE!")
print("="*60)
print(f"\nUse this threshold in your API: {RECOMMENDED_THRESHOLD:.2f}")
print(f"Save path: {save_dir}/threshold.pkl")
print("="*60)