import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import random

def create_sample_data(n_samples=10000):
    """
    Create sample bank customer data for churn prediction
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate sample data
    data = {
        'CustomerId': range(15000000, 15000000 + n_samples),
        'Surname': [f'Customer_{i}' for i in range(n_samples)],
        'CreditScore': np.random.normal(650, 96, n_samples).astype(int),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'Age': np.random.randint(18, 92, n_samples),
        'Tenure': np.random.randint(0, 11, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.46, 0.03, 0.01]),
        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
        'EstimatedSalary': np.random.uniform(11, 199992, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn patterns
    churn_prob = np.zeros(n_samples)
    
    # Age factor - older customers more likely to churn
    churn_prob += (df['Age'] - 40) * 0.01
    
    # Geography factor - Germany has higher churn
    churn_prob += np.where(df['Geography'] == 'Germany', 0.15, 0)
    churn_prob += np.where(df['Geography'] == 'Spain', 0.05, 0)
    
    # Gender factor - females slightly more likely to churn
    churn_prob += np.where(df['Gender'] == 'Female', 0.05, 0)
    
    # Balance factor - very high or very low balance increases churn
    churn_prob += np.where(df['Balance'] == 0, 0.2, 0)
    churn_prob += np.where(df['Balance'] > 200000, 0.1, 0)
    
    # Product factor - having only 1 or more than 2 products increases churn
    churn_prob += np.where(df['NumOfProducts'] == 1, 0.1, 0)
    churn_prob += np.where(df['NumOfProducts'] > 2, 0.2, 0)
    
    # Activity factor - inactive members more likely to churn
    churn_prob += np.where(df['IsActiveMember'] == 0, 0.15, 0)
    
    # Credit score factor - very low credit score increases churn
    churn_prob += np.where(df['CreditScore'] < 500, 0.1, 0)
    
    # Apply sigmoid to keep probabilities between 0 and 1
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    
    # Generate actual churn based on probabilities
    df['Exited'] = np.random.binomial(1, churn_prob)
    
    # Clip values to realistic ranges
    df['CreditScore'] = np.clip(df['CreditScore'], 350, 850)
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for machine learning
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Remove unnecessary columns
    features_to_drop = ['CustomerId', 'Surname']
    df_processed = df_processed.drop(columns=features_to_drop)
    
    # Encode categorical variables
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_processed['Geography'] = le_geography.fit_transform(df_processed['Geography'])
    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
    
    # Separate features and target
    X = df_processed.drop('Exited', axis=1)
    y = df_processed['Exited']
    
    return X, y, le_geography, le_gender

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def get_feature_names():
    """
    Get feature names for the dataset
    """
    return ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
