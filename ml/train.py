"""
UNCart - ML_29 Model Training & Evaluation Pipeline (Dual-Model Export)
- Primary Academic Model: Logistic Regression (Regularized L2) -> uncart_pipeline.joblib
- Benchmark Model: Random Forest Classifier -> uncart_rf_pipeline.joblib
- Target Performance: ~82-84% Accuracy, ~87-89% ROC-AUC
- Explainability: SHAP (LinearExplainer) for exact mathematical feature attribution
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import shap

def run_training_pipeline():
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(ml_dir, 'uncart_orders_dataset.csv')
    
    if not os.path.exists(dataset_path):
        from generate_dataset import generate_uncart_dataset
        print("Dataset not found. Generating dataset first...")
        df = generate_uncart_dataset()
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    print(f"\n[1/5] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Feature definition
    num_features = [
        'customer_age_days',
        'past_orders_count',
        'customer_past_return_rate',
        'product_price',
        'discount_percent',
        'item_quantity'
    ]
    
    cat_features = [
        'product_category',
        'payment_method',
        'delivery_tier',
        'is_festive_sale'
    ]
    
    target = 'is_returned'

    X = df[num_features + cat_features]
    y = df[target]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # Fit preprocessor on training data
    print("[2/5] Fitting preprocessor (StandardScaler + OneHotEncoder)...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names

    # ==========================================================
    # 1. Primary Model: Logistic Regression (Assigned ML_29)
    # ==========================================================
    print("[3/5] Training Primary Model: Logistic Regression (ML_29)...")
    log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    log_reg.fit(X_train_proc, y_train)

    y_pred_lr = log_reg.predict(X_test_proc)
    y_prob_lr = log_reg.predict_proba(X_test_proc)[:, 1]

    lr_metrics = {
        'model_name': 'Logistic Regression (ML_29)',
        'accuracy': round(float(accuracy_score(y_test, y_pred_lr)), 4),
        'precision': round(float(precision_score(y_test, y_pred_lr)), 4),
        'recall': round(float(recall_score(y_test, y_pred_lr)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred_lr)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob_lr)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    }

    # ==========================================================
    # 2. Benchmark Model: Random Forest
    # ==========================================================
    print("[4/5] Training Benchmark Model: Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_clf.fit(X_train_proc, y_train)

    y_pred_rf = rf_clf.predict(X_test_proc)
    y_prob_rf = rf_clf.predict_proba(X_test_proc)[:, 1]

    rf_metrics = {
        'model_name': 'Random Forest Classifier (Benchmark)',
        'accuracy': round(float(accuracy_score(y_test, y_pred_rf)), 4),
        'precision': round(float(precision_score(y_test, y_pred_rf)), 4),
        'recall': round(float(recall_score(y_test, y_pred_rf)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred_rf)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob_rf)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
    }

    # ==========================================================
    # 3. Explainable AI: SHAP Linear Explainer
    # ==========================================================
    print("[5/5] Calculating SHAP Linear Feature Attributions...")
    explainer = shap.LinearExplainer(
        log_reg, 
        X_train_proc, 
        feature_perturbation="interventional"
    )
    
    shap_sample = X_test_proc[:300]
    shap_values = explainer.shap_values(shap_sample)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = []
    for name, score in zip(all_feature_names, mean_abs_shap):
        shap_importance.append({
            'feature': name,
            'importance': round(float(score), 4)
        })
    shap_importance = sorted(shap_importance, key=lambda x: x['importance'], reverse=True)

    # ==========================================================
    # Save BOTH Production Pipelines (Logistic Reg + Random Forest)
    # ==========================================================
    # 1. Primary Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', log_reg)
    ])
    lr_pipeline_path = os.path.join(ml_dir, 'uncart_pipeline.joblib')
    joblib.dump(lr_pipeline, lr_pipeline_path)

    # 2. Benchmark Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])
    rf_pipeline_path = os.path.join(ml_dir, 'uncart_rf_pipeline.joblib')
    joblib.dump(rf_pipeline, rf_pipeline_path)

    # Save metrics and metadata
    metrics_path = os.path.join(ml_dir, 'model_metrics.json')
    shap_path = os.path.join(ml_dir, 'shap_summary.json')
    feature_names_path = os.path.join(ml_dir, 'feature_metadata.json')

    metrics_summary = {
        'primary_model': lr_metrics,
        'benchmark_model': rf_metrics,
        'features_used': all_feature_names,
        'test_set_size': len(y_test),
        'intercept': round(float(log_reg.intercept_[0]), 4)
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    with open(shap_path, 'w') as f:
        json.dump({'global_importance': shap_importance}, f, indent=2)

    with open(feature_names_path, 'w') as f:
        json.dump({
            'numerical': num_features,
            'categorical': cat_features,
            'encoded_all': all_feature_names
        }, f, indent=2)

    print("\n" + "="*55)
    print("           MODEL EVALUATION SUMMARY (ML_29)")
    print("="*55)
    print(f"Logistic Regression ROC-AUC : {lr_metrics['roc_auc'] * 100:.2f}%")
    print(f"Logistic Regression Accuracy: {lr_metrics['accuracy'] * 100:.2f}%")
    print(f"Logistic Regression Recall  : {lr_metrics['recall'] * 100:.2f}%")
    print("-"*55)
    print(f"Random Forest ROC-AUC       : {rf_metrics['roc_auc'] * 100:.2f}%")
    print(f"Random Forest Accuracy      : {rf_metrics['accuracy'] * 100:.2f}%")
    print("="*55)
    print("\nTop 5 Most Influential Features (SHAP Importance):")
    for i, item in enumerate(shap_importance[:5], 1):
        print(f"  {i}. {item['feature']} (impact: {item['importance']})")
        
    print(f"\n[SUCCESS] Both models exported successfully:")
    print(f"  1. Primary (Logistic Reg) : {lr_pipeline_path}")
    print(f"  2. Benchmark (Random Forest): {rf_pipeline_path}")

    return metrics_summary

if __name__ == '__main__':
    run_training_pipeline()
