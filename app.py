# requirements.txt:
# streamlit, pandas, numpy, scikit-learn, plotly, mlxtend, xgboost

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Machine Learning & Stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Association Rules
from mlxtend.frequent_patterns import apriori, association_rules

# Optional XGBoost handling
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Money Tree Bank | Fraud Analytics", page_icon="🌳")

# --- 2. CUSTOM THEMING (Professional Dark Banking) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00C896 !important; font-size: 32px; font-weight: bold; }
    .stSidebar { background-color: #161B22 !important; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #00C896; font-family: 'Inter', sans-serif; }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

ACCENT_COLOR = "#00C896"

# --- 3. DATA ENGINE ---
@st.cache_data
def load_and_preprocess_data():
    # Load raw data
    df = pd.read_csv("bank_transactions_data_2_augmented_clean_2.csv")
    
    # Date Parsing (handling mixed formats)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    
    # Handle Is_fraud (Fill NaN with 0, cast to int)
    if 'Is_fraud' in df.columns:
        df['Is_fraud'] = df['Is_fraud'].fillna(0).astype(int)
    else:
        df['Is_fraud'] = 0
        
    # Rule-Based Heuristics (Engineering Synthetic Fraud Labels)
    df['Fraud_Target'] = df['Is_fraud']
    # Condition 1: High attempts + Medium/High amount
    df.loc[(df['LoginAttempts'] >= 4) & (df['TransactionAmount'] > 700), 'Fraud_Target'] = 1
    # Condition 2: Low balance + High amount
    df.loc[(df['AccountBalance'] < 300) & (df['TransactionAmount'] > 500), 'Fraud_Target'] = 1
    # Condition 3: Unusually fast duration
    df.loc[df['TransactionDuration'] < 20, 'Fraud_Target'] = 1
    
    # Feature Engineering
    df['Hour'] = df['TransactionDate'].dt.hour
    df['MonthYear'] = df['TransactionDate'].dt.to_period('M').astype(str)
    
    return df

df_raw = load_and_preprocess_data()

# --- 4. GLOBAL SIDEBAR FILTERS ---
st.sidebar.markdown(f"<h2 style='text-align: center; color: {ACCENT_COLOR};'>🌳 Money Tree</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Filters
min_date = df_raw['TransactionDate'].min().date()
max_date = df_raw['TransactionDate'].max().date()
date_range = st.sidebar.date_input("🗓️ Date Range", [min_date, max_date])

channels = st.sidebar.multiselect("📡 Channel", options=df_raw['Channel'].unique(), default=df_raw['Channel'].unique())
occupations = st.sidebar.multiselect("💼 Occupation", options=df_raw['CustomerOccupation'].unique(), default=df_raw['CustomerOccupation'].unique())
txn_type = st.sidebar.radio("💳 Transaction Type", ["Both", "Credit", "Debit"])

# Apply Filters
df = df_raw.copy()
if len(date_range) == 2:
    df = df[(df['TransactionDate'].dt.date >= date_range[0]) & (df['TransactionDate'].dt.date <= date_range[1])]
df = df[df['Channel'].isin(channels)]
df = df[df['CustomerOccupation'].isin(occupations)]
if txn_type != "Both":
    df = df[df['TransactionType'] == txn_type]

# Navigation
page = st.sidebar.selectbox("📂 Dashboard Modules", 
    ["Home", "Dataset Overview", "EDA & Visualizations", "Classification Models", 
     "Clustering Analysis", "Association Rule Mining", "Regression Forecast", "Bias Detection"])

# --- PAGE: 1. HOME ---
if page == "Home":
    st.title("🏦 Fraud Detection Analytics Dashboard")
    st.markdown("---")
    
    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.write("""
        With the rapid growth of digital banking, financial institutions increasingly rely on data to detect suspicious activity and manage risk. 
        This dashboard analyzes 50,000 transaction records to uncover behavioral patterns, segment customers, and flag potential fraud using machine learning.
        
        **Key Strategic Questions addressed:**
        - Which transaction channels carry the highest fraud risk?
        - How do customer demographics relate to transaction behavior?
        - What patterns distinguish fraudulent from legitimate transactions?
        """)
    with col_img:
        st.image("https://img.icons8.com/fluency/200/safe-ok.png")

    st.markdown("### 📊 Portfolio Key Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Transactions", f"{len(df):,}")
    m2.metric("Total Volume", f"₹{df['TransactionAmount'].sum()/1e6:.2f}M")
    m3.metric("Avg Transaction", f"₹{df['TransactionAmount'].mean():.2f}")
    m4.metric("Fraud Rate", f"{(df['Fraud_Target'].mean()*100):.2f}%")
    m5.metric("High Login Alerts", f"{len(df[df['LoginAttempts'] >= 3]):,}")

# --- PAGE: 2. DATASET OVERVIEW ---
elif page == "Dataset Overview":
    st.header("📋 Dataset Intelligence")
    st.dataframe(df.head(100), use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    st.subheader("Statistical Profile")
    st.write(df.describe())
    st.caption("Insight: Wide distribution in account balances suggests a need for targeted cluster risk analysis.")

# --- PAGE: 3. EDA & VISUALIZATIONS ---
elif page == "EDA & Visualizations":
    st.header("📈 Behavioral Trends")
    template = "plotly_dark"

    row1_1, row1_2 = st.columns(2)
    with row1_1:
        time_data = df.groupby('MonthYear').size().reset_index()
        time_data.columns = ['Month', 'Count']
        st.plotly_chart(px.line(time_data, x='Month', y='Count', title="Volume Over Time", template=template, color_discrete_sequence=[ACCENT_COLOR]), use_container_width=True)
    
    with row1_2:
        st.plotly_chart(px.histogram(df, x='Hour', nbins=24, title="Hourly Activity Patterns", template=template, color_discrete_sequence=[ACCENT_COLOR]), use_container_width=True)

    row2_1, row2_2 = st.columns(2)
    with row2_1:
        chan_data = df['Channel'].value_counts().reset_index()
        chan_data.columns = ['Channel', 'Count'] # Version-proof fix
        st.plotly_chart(px.bar(chan_data, x='Channel', y='Count', title="Channel Distribution", template=template, color_discrete_sequence=[ACCENT_COLOR]), use_container_width=True)
    
    with row2_2:
        st.plotly_chart(px.pie(df, names='TransactionType', title="Credit vs Debit Ratio", hole=0.5, template=template, color_discrete_sequence=[ACCENT_COLOR, '#008262']), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(px.box(df, x='CustomerOccupation', y='TransactionAmount', color='CustomerOccupation', title="Amount Spread by Occupation", template=template), use_container_width=True)

# --- PAGE: 4. CLASSIFICATION MODELS ---
elif page == "Classification Models":
    st.header("🤖 Machine Learning Fraud Classification")
    
    with st.spinner("Training Models..."):
        le = LabelEncoder()
        df_ml = df.copy()
        for col in ['Channel', 'TransactionType', 'CustomerOccupation']:
            df_ml[col] = le.fit_transform(df_ml[col])
        
        features = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance', 'Channel', 'TransactionType', 'CustomerOccupation']
        X = df_ml[features]
        y = df_ml['Fraud_Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }
        if XGB_AVAILABLE: models["XGBoost"] = XGBClassifier(eval_metric='logloss')

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            p = model.predict(X_test)
            results.append({"Model": name, "F1": f1_score(y_test, p), "Recall": recall_score(y_test, p), "Precision": precision_score(y_test, p)})

    st.table(pd.DataFrame(results))
    
    # Feature Importance from Random Forest
    rf = models["Random Forest"]
    fi = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values('Importance')
    st.plotly_chart(px.bar(fi, x='Importance', y='Feature', orientation='h', title="Primary Fraud Indicators", template="plotly_dark", color_discrete_sequence=[ACCENT_COLOR]), use_container_width=True)
    st.info("Insight: Login Attempts and Transaction Duration are the strongest predictors of fraudulent intent.")

# --- PAGE: 5. CLUSTERING ANALYSIS ---
elif page == "Clustering Analysis":
    st.header("🧬 Customer Segmentation")
    X_sc = StandardScaler().fit_transform(df[['TransactionAmount', 'AccountBalance', 'LoginAttempts']])
    
    # Simple elbow for visual
    km = KMeans(n_clusters=4).fit(X_sc)
    df['Cluster'] = km.labels_
    
    pca = PCA(n_components=2).fit_transform(X_sc)
    df['PCA1'], df['PCA2'] = pca[:,0], pca[:,1]
    
    st.plotly_chart(px.scatter(df, x='PCA1', y='PCA2', color='Cluster', title="Behavioral Clusters (PCA)", template="plotly_dark"), use_container_width=True)
    st.write(df.groupby('Cluster')[['TransactionAmount', 'AccountBalance', 'LoginAttempts']].mean())

# --- PAGE: 6. ASSOCIATION RULE MINING ---
elif page == "Association Rule Mining":
    st.header("🔗 Pattern Discovery (Apriori)")
    df_assoc = pd.get_dummies(df[['Channel', 'TransactionType', 'CustomerOccupation', 'Fraud_Target']])
    freq = apriori(df_assoc, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1)
    st.dataframe(rules.sort_values('lift', ascending=False).head(10), use_container_width=True)

# --- PAGE: 7. REGRESSION FORECAST ---
elif page == "Regression Forecast":
    st.header("📉 Transaction Value Forecast")
    X_reg = df[['CustomerAge', 'AccountBalance', 'LoginAttempts']]
    y_reg = df['TransactionAmount']
    reg = LinearRegression().fit(X_reg, y_reg)
    df['Pred'] = reg.predict(X_reg)
    
    st.plotly_chart(px.scatter(df.head(500), x='TransactionAmount', y='Pred', trendline="ols", title="Actual vs Predicted Values", template="plotly_dark"), use_container_width=True)
    st.metric("Model R² Score", f"{r2_score(y_reg, df['Pred']):.4f}")

# --- PAGE: 8. BIAS DETECTION ---
elif page == "Bias Detection":
    st.header("⚖️ Fairness Analysis")
    bias = df.groupby('CustomerOccupation')['Fraud_Target'].mean().reset_index()
    st.plotly_chart(px.bar(bias, x='CustomerOccupation', y='Fraud_Target', title="Fraud Exposure by Occupation", template="plotly_dark", color_discrete_sequence=[ACCENT_COLOR]), use_container_width=True)
    
    disparity = bias['Fraud_Target'].max() / (bias['Fraud_Target'].min() + 0.0001)
    st.metric("Disparity Ratio", f"{disparity:.2f}x")
    if disparity > 1.5: st.warning("Bias Warning: Certain occupations show significantly higher fraud flags.")
