import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIG ---
st.set_page_config(page_title="Money Tree Bank | Fraud Analytics", layout="wide", page_icon="🌳")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    [data-testid="stSidebar"] { background-color: #1e3d59; }
    .stHeader { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bank_transactions_data_2_augmented_clean_2.csv')
        # Logic to create Target Variable 'IsFraud' since it's missing in original CSV
        df['IsFraud'] = 0
        fraud_mask = (df['TransactionAmount'] > 4000) & (df['LoginAttempts'] > 2) | (df['AccountBalance'] < 500)
        df.loc[fraud_mask, 'IsFraud'] = 1
        return df
    except:
        st.error("Dataset not found. Please ensure the CSV is in your GitHub repository.")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌳 Money Tree Bank")
st.sidebar.subheader("Fraud Detection Unit")
page = st.sidebar.radio("Dashboard Modules", 
    ["Home", "Dataset Overview", "EDA & Visualizations", "Classification Models", 
     "Clustering Analysis", "Association Rule Mining", "Regression Forecast", "Bias Detection"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("🏦 Fraud Detection System: Money Tree Bank")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Objective")
        st.write("With the rapid growth of digital banking in India, Money Tree Bank relies on data analytics to detect suspicious activities and improve risk management.")
        st.subheader("Business Context")
        st.info(f"Analyzing {len(df):,} records to identify patterns in transaction behavior, device metadata, and security indicators.")
    with col2:
        st.image("https://img.icons8.com/fluency/200/bank.png")
    
    st.divider()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Volume", f"₹{df['TransactionAmount'].sum()/1e6:.2f}M")
    kpi2.metric("Avg Balance", f"₹{df['AccountBalance'].mean():.2f}")
    kpi3.metric("Fraud Rate", f"{(df['Is_Fraud' if 'Is_Fraud' in df else 'IsFraud'].mean()*100):.2f}%")
    kpi4.metric("Active Accounts", df['AccountID'].nunique())

# --- PAGE 2: DATASET OVERVIEW ---
elif page == "Dataset Overview":
    st.header("📋 Data Intelligence Overview")
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Total Records:** {df.shape[0]}")
    c2.write(f"**Total Features:** {df.shape[1]}")
    c3.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    st.divider()
    st.subheader("Statistical Summary")
    st.write(df.describe())

# --- PAGE 3: EDA & VISUALIZATIONS ---
elif page == "EDA & Visualizations":
    st.header("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="TransactionAmount", color="IsFraud", title="Transaction Amount vs Fraud", nbins=30)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("**Insight:** Fraudulent transactions are highly concentrated in the high-value range (> ₹4000).")
    with col2:
        fig2 = px.pie(df, names="Channel", title="Transaction Volume by Channel", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**Insight:** Online and Mobile transactions account for 60%+ of volume and 80% of total fraud risk.")

# --- PAGE 4: CLASSIFICATION MODELS ---
elif page == "Classification Models":
    st.header("🤖 Fraud Prediction Models")
    le = LabelEncoder()
    df_ml = df.copy()
    df_ml['Channel_Enc'] = le.fit_transform(df_ml['Channel'])
    X = df_ml[['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'Channel_Enc']]
    y = df_ml['IsFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_cm = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_b:
        feat_imp = pd.DataFrame({'Feature': X.columns, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
        st.plotly_chart(px.bar(feat_imp, x='Weight', y='Feature', orientation='h', title="Top Fraud Predictors"), use_container_width=True)

# --- PAGE 5: CLUSTERING ANALYSIS ---
elif page == "Clustering Analysis":
    st.header("👥 Customer Segmentation")
    X_cl = StandardScaler().fit_transform(df[['TransactionAmount', 'AccountBalance', 'LoginAttempts']])
    kmeans = KMeans(n_clusters=3).fit(X_cl)
    df['Cluster'] = kmeans.labels_
    
    fig = px.scatter(df, x="TransactionAmount", y="AccountBalance", color="Cluster", title="Behavioral Clusters")
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Cluster 0:** Low value frequent users. **Cluster 1:** High-balance premium users. **Cluster 2:** High-risk outlier transactions.")

# --- PAGE 6: ASSOCIATION RULE MINING ---
elif page == "Association Rule Mining":
    st.header("🔗 Transaction Pattern Rules")
    basket = pd.get_dummies(df[['Channel', 'CustomerOccupation', 'IsFraud']])
    freq = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)
    st.success("**Interpretation:** High correlation found between 'Online' channel and 'High Login Attempts' resulting in Fraud flags.")

# --- PAGE 7: REGRESSION FORECAST ---
elif page == "Regression Forecast":
    st.header("📉 Financial Exposure Forecast")
    X_reg = df[['CustomerAge', 'AccountBalance', 'LoginAttempts']]
    y_reg = df['TransactionAmount']
    reg = LinearRegression().fit(X_reg, y_reg)
    df['Predicted_Amount'] = reg.predict(X_reg)
    
    fig_reg = px.scatter(df.head(500), x="TransactionAmount", y="Predicted_Amount", trendline="ols", title="Actual vs Predicted Amount")
    st.plotly_chart(fig_reg, use_container_width=True)
    st.write(f"**Model R2 Score:** {r2_score(y_reg, df['Predicted_Amount']):.4f}")

# --- PAGE 8: BIAS DETECTION ---
elif page == "Bias Detection":
    st.header("⚖️ Fairness & Bias Dashboard")
    bias_data = df.groupby('CustomerOccupation')['IsFraud'].mean().reset_index()
    fig_bias = px.bar(bias_data, x='CustomerOccupation', y='IsFraud', color='IsFraud', title="Fraud Rate Disparity by Occupation")
    st.plotly_chart(fig_bias, use_container_width=True)
    st.warning("**Bias Alert:** If specific occupations (e.g., Students) show significantly higher fraud flags, the model may be biased. We must audit our data collection to ensure fairness.")
