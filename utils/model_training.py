import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def train_multiple_models(X_train, y_train):
    """
    Train multiple machine learning models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and return metrics
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    return results

def get_model_predictions(model, X_test):
    """
    Get predictions and probabilities from a model
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    return predictions, probabilities

def print_model_evaluation(models, X_test, y_test):
    """
    Print detailed evaluation for all models
    """
    results = evaluate_models(models, X_test, y_test)
    
    print("="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Create results dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print(results_df)
    print("\n")
    
    # Find best model for each metric
    print("BEST MODELS BY METRIC:")
    print("-" * 40)
    for metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df.loc[best_model, metric]
        print(f"{metric.upper()}: {best_model} ({best_score:.4f})")
    
    print("\n")
    return results_df

def get_feature_importance(model, feature_names):
    """
    Get feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        print("This model doesn't support feature importance.")
        return None

def predict_single_customer(model, scaler, customer_data):
    """
    Predict churn for a single customer
    
    customer_data should be a dictionary with keys:
    ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
     'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    """
    # Convert to array
    features = np.array(list(customer_data.values())).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def create_confusion_matrix_data(y_true, y_pred):
    """
    Create confusion matrix data for visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'True Negative': cm[0, 0],
        'False Positive': cm[0, 1], 
        'False Negative': cm[1, 0],
        'True Positive': cm[1, 1]
    }

def calculate_business_metrics(y_true, y_pred, customer_value=1000, retention_cost=100):
    """
    Calculate business-relevant metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        customer_value: Average customer lifetime value
        retention_cost: Cost to retain a customer
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business calculations
    customers_saved = tp  # True positives - correctly identified churners
    false_alarms = fp     # False positives - incorrectly identified as churners
    missed_churners = fn  # False negatives - missed churners
    
    # Financial impact
    revenue_saved = customers_saved * customer_value
    unnecessary_costs = false_alarms * retention_cost
    lost_revenue = missed_churners * customer_value
    
    net_benefit = revenue_saved - unnecessary_costs
    
    return {
        'customers_saved': customers_saved,
        'false_alarms': false_alarms,
        'missed_churners': missed_churners,
        'revenue_saved': revenue_saved,
        'unnecessary_costs': unnecessary_costs,
        'lost_revenue': lost_revenue,
        'net_benefit': net_benefit
    }
