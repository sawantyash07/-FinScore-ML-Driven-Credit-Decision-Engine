import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Page Config
st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

st.title("🏦 Loan Approval Analysis Dashboard")
st.markdown("This application analyzes loan applicant data and visualizes key insights.")

# 1. Load the dataset
@st.cache_data
def load_data():
    # Replace with your actual file path if different
    df = pd.read_csv('loan_approval_data.csv')
    return df

try:
    df_raw = load_data()
    df = df_raw.copy()

    # 2. Sidebar - Data Overview
    st.sidebar.header("Data Settings")
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Dataset")
        st.write(df.head())

    # 3. Data Cleaning (Handling Missing Values)
    # Identifying columns based on your script logic
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # 4. Layout: Metrics and Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Balance")
        counts = df['Loan_Approved'].value_counts()
        
        # Pie Chart
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                   colors=['skyblue', 'salmon'], startangle=140)
        ax_pie.axis('equal') 
        st.pyplot(fig_pie)

    with col2:
        st.subheader("Income Distribution")
        # Histogram for Applicant Income
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df['Applicant_Income'], kde=True, color='purple', ax=ax_hist)
        ax_hist.set_xlabel('Applicant Income')
        st.pyplot(fig_hist)

    # 5. Data Summary
    st.subheader("Dataset Statistics")
    st.write(df.describe())

except FileNotFoundError:
    st.error("Error: 'loan_approval_data.csv' not found. Please ensure the CSV file is in the same directory.")