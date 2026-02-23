# src/model_training.py (COMPLETE IMPROVED VERSION)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENHANCED MODEL TRAINING WITH ACCURACY IMPROVEMENTS")
print("="*60)

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("\n" + "="*60)
print("STEP 1: Loading Data")
print("="*60)

df = pd.read_csv('data/heart_disease_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nTarget distribution:")
print(df['heart_disease'].value_counts())
print(f"\nClass percentages:")
print(df['heart_disease'].value_counts(normalize=True).mul(100).round(2))

# ============================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ============================================
print("\n" + "="*60)
print("STEP 2: Separating Features and Target")
print("="*60)

X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================
print("\n" + "="*60)
print("STEP 3: Train-Test Split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test distribution:\n{y_test.value_counts(normalize=True)}")

# ============================================
# STEP 4: FEATURE SCALING
# ============================================
print("\n" + "="*60)
print("STEP 4: Feature Scaling")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Feature scaling complete!")
print(f"Training data mean (should be ~0): {X_train_scaled.mean().mean():.4f}")
print(f"Training data std (should be ~1): {X_train_scaled.std().mean():.4f}")

# ============================================
# STEP 5: HANDLE CLASS IMBALANCE
# ============================================
print("\n" + "="*60)
print("STEP 5: Handling Class Imbalance")
print("="*60)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights - Class 0: {class_weights[0]:.3f}, Class 1: {class_weights[1]:.3f}")

# ============================================
# STEP 6: CROSS-VALIDATION SETUP
# ============================================
print("\n" + "="*60)
print("STEP 6: Cross-Validation Setup")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Basic models for comparison
basic_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

print("\nCross-Validation Results (5-fold):")
cv_results = []
for name, model in basic_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall')
    cv_results.append({
        'Model': name,
        'CV Recall Mean': cv_scores.mean(),
        'CV Recall Std': cv_scores.std()
    })
    print(f"{name:20} CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================
# STEP 7: FEATURE ENGINEERING
# ============================================
print("\n" + "="*60)
print("STEP 7: Feature Engineering")
print("="*60)

# Create interaction features for important medical features
important_features = ['age', 'cholesterol', 'max_heart_rate', 'resting_blood_pressure']
print(f"Creating interaction features for: {important_features}")

# Polynomial features (interaction only)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[important_features])

# Create new feature names
interaction_names = []
feature_pairs = []
for i, feat1 in enumerate(important_features):
    for j, feat2 in enumerate(important_features):
        if i <= j:  # Include self-interactions
            interaction_names.append(f"{feat1}_x_{feat2}")
            feature_pairs.append((feat1, feat2))

# Combine with original features
X_enhanced = np.hstack([X.values, X_poly])
feature_names = list(X.columns) + interaction_names

print(f"Original features: {len(X.columns)}")
print(f"Added interaction features: {len(interaction_names)}")
print(f"Total features: {len(feature_names)}")

# Split enhanced features
X_train_enhanced = X_enhanced[:len(X_train)]
X_test_enhanced = X_enhanced[len(X_train):]

# Scale enhanced features
scaler_enhanced = StandardScaler()
X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)

# ============================================
# STEP 8: HYPERPARAMETER TUNING FOR RANDOM FOREST
# ============================================
print("\n" + "="*60)
print("STEP 8: Hyperparameter Tuning for Random Forest")
print("="*60)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

# Create base model
rf = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
print("Training Random Forest with GridSearchCV (this may take a few minutes)...")
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,  # Using 3-fold for speed
    scoring='recall',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Recall: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# ============================================
# STEP 9: TRAIN MODELS WITH CLASS WEIGHTS
# ============================================
print("\n" + "="*60)
print("STEP 9: Training Models with Class Weights")
print("="*60)

# Models with class weights
weighted_models = {
    'Logistic Regression (Balanced)': LogisticRegression(
        random_state=42, 
        class_weight='balanced',
        max_iter=1000
    ),
    'Random Forest (Tuned)': best_rf,
    'SVM (Balanced)': SVC(
        random_state=42, 
        probability=True,
        class_weight='balanced'
    )
}

# Train and evaluate each model
results = {}
for name, model in weighted_models.items():
    print(f"\n{'-'*50}")
    print(f"Training {name}...")
    print(f"{'-'*50}")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Store results
    results[name] = {
        'Model': model,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc
    }
    
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# ============================================
# STEP 10: ENSEMBLE METHODS
# ============================================
print("\n" + "="*60)
print("STEP 10: Ensemble Methods")
print("="*60)

# 1. Voting Classifier
print("\nTraining Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
y_proba_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]

recall_voting = recall_score(y_test, y_pred_voting)
precision_voting = precision_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
roc_auc_voting = roc_auc_score(y_test, y_proba_voting)

results['Voting Classifier'] = {
    'Model': voting_clf,
    'Recall': recall_voting,
    'Precision': precision_voting,
    'F1-Score': f1_voting,
    'Accuracy': accuracy_voting,
    'ROC-AUC': roc_auc_voting
}

print(f"Voting Classifier Results:")
print(f"  Recall: {recall_voting:.4f}")
print(f"  Precision: {precision_voting:.4f}")
print(f"  F1-Score: {f1_voting:.4f}")

# 2. Train on Enhanced Features
print("\nTraining Random Forest on Enhanced Features...")
rf_enhanced = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf_enhanced.fit(X_train_enhanced_scaled, y_train)

y_pred_enhanced = rf_enhanced.predict(X_test_enhanced_scaled)
y_proba_enhanced = rf_enhanced.predict_proba(X_test_enhanced_scaled)[:, 1]

recall_enhanced = recall_score(y_test, y_pred_enhanced)
precision_enhanced = precision_score(y_test, y_pred_enhanced)
f1_enhanced = f1_score(y_test, y_pred_enhanced)
accuracy_enhanced = accuracy_score(y_test, y_pred_enhanced)
roc_auc_enhanced = roc_auc_score(y_test, y_proba_enhanced)

results['Random Forest (Enhanced Features)'] = {
    'Model': rf_enhanced,
    'Recall': recall_enhanced,
    'Precision': precision_enhanced,
    'F1-Score': f1_enhanced,
    'Accuracy': accuracy_enhanced,
    'ROC-AUC': roc_auc_enhanced
}

print(f"Enhanced Features Results:")
print(f"  Recall: {recall_enhanced:.4f}")
print(f"  Precision: {precision_enhanced:.4f}")
print(f"  F1-Score: {f1_enhanced:.4f}")

# ============================================
# STEP 11: FEATURE SELECTION
# ============================================
print("\n" + "="*60)
print("STEP 11: Feature Selection")
print("="*60)

# Get feature importance from best Random Forest
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train_scaled, y_train)

# Display feature importance
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance Ranking:")
for i in range(len(X.columns)):
    print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Select top features (those above median importance)
selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()].tolist()
print(f"\nSelected {len(selected_features)} features: {selected_features}")

# Train on selected features
rf_selected = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf_selected.fit(X_train_selected, y_train)

y_pred_selected = rf_selected.predict(X_test_selected)
y_proba_selected = rf_selected.predict_proba(X_test_selected)[:, 1]

recall_selected = recall_score(y_test, y_pred_selected)
precision_selected = precision_score(y_test, y_pred_selected)
f1_selected = f1_score(y_test, y_pred_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
roc_auc_selected = roc_auc_score(y_test, y_proba_selected)

results['Random Forest (Feature Selection)'] = {
    'Model': rf_selected,
    'Recall': recall_selected,
    'Precision': precision_selected,
    'F1-Score': f1_selected,
    'Accuracy': accuracy_selected,
    'ROC-AUC': roc_auc_selected
}

print(f"Feature Selection Results:")
print(f"  Recall: {recall_selected:.4f}")
print(f"  Precision: {precision_selected:.4f}")

# ============================================
# STEP 12: FINAL MODEL COMPARISON
# ============================================
print("\n" + "="*60)
print("STEP 12: Final Model Comparison")
print("="*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    name: {
        'Recall': metrics['Recall'],
        'Precision': metrics['Precision'],
        'F1-Score': metrics['F1-Score'],
        'Accuracy': metrics['Accuracy'],
        'ROC-AUC': metrics['ROC-AUC']
    }
    for name, metrics in results.items()
}).T

# Sort by F1-Score (balance between Recall and Precision)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*60)
print("FINAL MODEL COMPARISON (sorted by F1-Score):")
print("="*60)
print(comparison_df.round(4).to_string())

# Find best model based on F1-Score
best_model_name = comparison_df.index[0]
best_model = results[best_model_name]['Model']
best_f1 = comparison_df.iloc[0]['F1-Score']
best_recall = comparison_df.iloc[0]['Recall']
best_precision = comparison_df.iloc[0]['Precision']

print(f"\n{'★'*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"  Recall:    {best_recall:.4f}")
print(f"  Precision: {best_precision:.4f}")
print(f"  F1-Score:  {best_f1:.4f}")
print(f"{'★'*60}")

# ============================================
# STEP 13: SAVE ALL MODELS AND ARTIFACTS
# ============================================
print("\n" + "="*60)
print("STEP 13: Saving Models and Artifacts")
print("="*60)

# Create directories
os.makedirs('models/enhanced', exist_ok=True)

# Save best model
joblib.dump(best_model, 'models/enhanced/best_model.pkl')
print(f"✅ Best model saved to models/enhanced/best_model.pkl")

# Save scalers
joblib.dump(scaler, 'models/enhanced/scaler.pkl')
joblib.dump(scaler_enhanced, 'models/enhanced/scaler_enhanced.pkl')
print(f"✅ Scalers saved")

# Save feature names
feature_info = {
    'original_features': list(X.columns),
    'selected_features': selected_features if 'selected_features' in locals() else list(X.columns),
    'feature_importance': dict(zip(X.columns, importances))
}
joblib.dump(feature_info, 'models/enhanced/feature_info.pkl')
print(f"✅ Feature information saved")

# Save comparison results
comparison_df.to_csv('models/enhanced/model_comparison.csv')
print(f"✅ Model comparison saved to models/enhanced/model_comparison.csv")

# Save training summary
with open('models/enhanced/training_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Recall: {best_recall:.4f}\n")
    f.write(f"Precision: {best_precision:.4f}\n")
    f.write(f"F1-Score: {best_f1:.4f}\n\n")
    f.write("Model Comparison:\n")
    f.write(comparison_df.round(4).to_string())

print(f"✅ Training summary saved")

# ============================================
# STEP 14: SAVE PROCESSED DATA
# ============================================
print("\n" + "="*60)
print("STEP 14: Saving Processed Data")
print("="*60)

# Save scaled data
os.makedirs('data/processed/enhanced', exist_ok=True)
X_train_scaled.to_csv('data/processed/enhanced/X_train.csv', index=False)
X_test_scaled.to_csv('data/processed/enhanced/X_test.csv', index=False)
y_train.to_csv('data/processed/enhanced/y_train.csv', index=False)
y_test.to_csv('data/processed/enhanced/y_test.csv', index=False)

# Save enhanced features if they exist
if 'X_train_enhanced_scaled' in locals():
    pd.DataFrame(X_train_enhanced_scaled).to_csv('data/processed/enhanced/X_train_enhanced.csv', index=False)
    pd.DataFrame(X_test_enhanced_scaled).to_csv('data/processed/enhanced/X_test_enhanced.csv', index=False)

print(f"✅ Processed data saved to data/processed/enhanced/")

print("\n" + "="*60)
print("✅ ENHANCED MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {best_f1:.4f} (Balance of Recall & Precision)")
print(f"Recall: {best_recall:.4f}")
print(f"Precision: {best_precision:.4f}")
print("\nNext steps:")
print("1. Run threshold optimization: python src/optimize_threshold_enhanced.py")
print("2. Update API to use enhanced model")
print("3. Test with Streamlit app")
print("="*60)