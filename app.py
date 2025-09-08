import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import create_sample_data, preprocess_data, scale_features
from utils.model_training import train_multiple_models, evaluate_models
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .churn-no {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare data for modeling"""
    df = create_sample_data()
    X, y, le_geography, le_gender = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return df, X_train_scaled, X_test_scaled, y_train, y_test, le_geography, le_gender, scaler

@st.cache_resource
def train_models():
    """Train and cache models"""
    df, X_train_scaled, X_test_scaled, y_train, y_test, le_geography, le_gender, scaler = load_and_prepare_data()
    models = train_multiple_models(X_train_scaled, y_train)
    results = evaluate_models(models, X_test_scaled, y_test)
    return models, results, scaler, le_geography, le_gender

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🏦 Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Predict customer churn using machine learning models")
    
    # Load data and models
    df, X_train_scaled, X_test_scaled, y_train, y_test, le_geography, le_gender, scaler = load_and_prepare_data()
    models, results, trained_scaler, trained_le_geography, trained_le_gender = train_models()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", 
                           ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Prediction", "📈 Model Comparison"])
    
    if page == "🏠 Home":
        home_page(df)
    elif page == "📊 Data Analysis":
        data_analysis_page(df)
    elif page == "🤖 Model Training":
        model_training_page(results)
    elif page == "🔮 Prediction":
        prediction_page(models, trained_scaler, trained_le_geography, trained_le_gender)
    elif page == "📈 Model Comparison":
        model_comparison_page(results)

def home_page(df):
    """Home page with overview"""
    st.markdown('<h2 class="sub-header">📋 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Churned Customers", df['Exited'].sum())
    with col3:
        churn_rate = (df['Exited'].sum() / len(df)) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("### 🎯 Project Objectives")
    st.write("""
    - **Predict Customer Churn**: Identify customers likely to leave the bank
    - **Business Impact**: Reduce customer acquisition costs and improve retention
    - **Model Comparison**: Compare different machine learning algorithms
    - **Interactive Prediction**: Real-time churn prediction for new customers
    """)
    
    if st.checkbox("Show Sample Data"):
        st.dataframe(df.head(10))

def data_analysis_page(df):
    """Data analysis and visualization page"""
    st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
    with col2:
        st.subheader("Target Distribution")
        churn_counts = df['Exited'].value_counts()
        fig = px.pie(values=churn_counts.values, names=['Retained', 'Churned'], 
                     title="Customer Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Age distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Age', color='Exited', nbins=20, 
                          title='Age Distribution by Churn Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='Exited', y='Balance', 
                     title='Balance Distribution by Churn Status')
        st.plotly_chart(fig, use_container_width=True)
    
    # Geography and Gender analysis
    col1, col2 = st.columns(2)
    
    with col1:
        geography_churn = df.groupby(['Geography', 'Exited']).size().unstack()
        fig = px.bar(geography_churn, title='Churn by Geography')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_churn = df.groupby(['Gender', 'Exited']).size().unstack()
        fig = px.bar(gender_churn, title='Churn by Gender')
        st.plotly_chart(fig, use_container_width=True)

def model_training_page(results):
    """Model training results page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training Results</h2>', unsafe_allow_html=True)
    
    # Display model performance
    results_df = pd.DataFrame(results).T
    st.subheader("Model Performance Comparison")
    st.dataframe(results_df.round(4))
    
    # Best model identification
    best_model = results_df['accuracy'].idxmax()
    st.success(f"🏆 Best Model: **{best_model}** with accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")
    
    # Visualize model comparison
    fig = px.bar(results_df.reset_index(), x='index', y=['accuracy', 'precision', 'recall', 'f1'], 
                 title='Model Performance Metrics Comparison', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def prediction_page(models, scaler, le_geography, le_gender):
    """Interactive prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Customer Churn Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
    
    with col2:
        st.subheader("Account Details")
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card", ['Yes', 'No'])
        is_active = st.selectbox("Is Active Member", ['Yes', 'No'])
        salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)
    
    # Process inputs - ensure all are numeric
    try:
        geography_encoded = int(le_geography.transform([geography])[0])
        gender_encoded = int(le_gender.transform([gender]))
        has_cr_card_encoded = 1 if has_cr_card == 'Yes' else 0
        is_active_encoded = 1 if is_active == 'Yes' else 0
        
        # Create input array with explicit type conversion
        input_features = [
            float(credit_score),
            float(geography_encoded),
            float(gender_encoded), 
            float(age),
            float(tenure),
            float(balance),
            float(num_products),
            float(has_cr_card_encoded),
            float(is_active_encoded),
            float(salary)
        ]
        
        # Create 2D array for sklearn (1 sample, n features)
        input_data = np.array(input_features).reshape(1, -1)
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
    except Exception as e:
        st.error(f"Error processing input data: {e}")
        return
    
    # Model selection
    st.subheader("Model Selection")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    
    if st.button("🔮 Predict Churn", type="primary"):
        try:
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.markdown(f'<div class="prediction-result churn-yes">⚠️ Customer is likely to CHURN</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-result churn-no">✅ Customer is likely to STAY</div>', 
                           unsafe_allow_html=True)
            
            # Probability visualization
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{probability[1]:.2%}")
            with col2:
                st.metric("Retention Probability", f"{probability:.2%}")
            
            # Simple progress bar instead of complex gauge
            st.write("**Churn Risk Level:**")
            st.progress(probability[1])
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Please check your input values and try again.")
       

def model_comparison_page(results):
    """Detailed model comparison page"""
    st.markdown('<h2 class="sub-header">📈 Detailed Model Comparison</h2>', unsafe_allow_html=True)
    
    results_df = pd.DataFrame(results).T
    
    # Radar chart for model comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    fig = go.Figure()
    
    for model in results_df.index:
        fig.add_trace(go.Scatterpolar(
            r=results_df.loc[model, metrics].values,
            theta=metrics,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    st.dataframe(results_df.round(4))
    
    # Model recommendations
    st.subheader("Model Recommendations")
    best_accuracy = results_df['accuracy'].idxmax()
    best_precision = results_df['precision'].idxmax()
    best_recall = results_df['recall'].idxmax()
    best_f1 = results_df['f1'].idxmax()
    
    st.write(f"**Best Accuracy:** {best_accuracy} ({results_df.loc[best_accuracy, 'accuracy']:.4f})")
    st.write(f"**Best Precision:** {best_precision} ({results_df.loc[best_precision, 'precision']:.4f})")
    st.write(f"**Best Recall:** {best_recall} ({results_df.loc[best_recall, 'recall']:.4f})")
    st.write(f"**Best F1-Score:** {best_f1} ({results_df.loc[best_f1, 'f1']:.4f})")

if __name__ == "__main__":
    main()
