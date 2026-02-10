import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(
    page_title="Loan Insight Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetricValue"] { font-size: 28px; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Data Loading & Model Training
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('loan_approval_data.csv')
        
        # Clean Data
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        
        # Prepare a simple model for prediction
        # We pick key features likely present in your CSV
        features = ['Applicant_Income', 'Credit_Score', 'Age', 'Existing_Loans']
        available_features = [f for f in features if f in df.columns]
        
        X = df[available_features]
        # Convert target to 1 and 0
        y = df['Loan_Approved'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
        model = LogisticRegression()
        model.fit(X, y)
        
        return df, model, available_features
    except Exception as e:
        return None, None, None

df, model, feature_list = load_and_prepare_data()

if df is not None:
    # 3. Sidebar Filters
    st.sidebar.title("🎮 Dashboard Controls")
    if 'Employment_Status' in df.columns:
        options = df['Employment_Status'].unique().tolist()
        emp_status = st.sidebar.multiselect("Filter by Employment", options, default=options)
        df_filtered = df[df['Employment_Status'].isin(emp_status)]
    else:
        df_filtered = df

    # 4. Main Tabs
    tab_dashboard, tab_predictor, tab_data = st.tabs(["📊 Analytics Dashboard", "🚀 Loan Predictor", "📋 Raw Data"])

    # --- TAB 1: DASHBOARD ---
    with tab_dashboard:
        st.title("💰 Loan Approval Analytics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", f"{len(df_filtered):,}")
        with col2:
            st.metric("Avg. Income", f"${df_filtered['Applicant_Income'].mean():,.0f}")
        with col3:
            rate = (df_filtered['Loan_Approved'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).mean() * 100)
            st.metric("Approval Rate", f"{rate:.1f}%")
        with col4:
            st.metric("Avg. Credit Score", f"{int(df_filtered['Credit_Score'].mean())}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df_filtered, names='Loan_Approved', hole=0.5, title="Approval Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(df_filtered, x='Applicant_Income', y='Credit_Score', color='Loan_Approved', title="Income vs Credit Score")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 2: PREDICTOR (NEW FEATURE) ---
    with tab_predictor:
        st.title("🚀 Real-Time Loan Eligibility")
        st.markdown("Enter applicant details below to predict loan approval.")
        
        with st.container():
            col_a, col_b = st.columns(2)
            user_input = {}
            
            # Dynamically create inputs based on the features the model uses
            for i, feat in enumerate(feature_list):
                with col_a if i % 2 == 0 else col_b:
                    user_input[feat] = st.number_input(f"Enter {feat.replace('_', ' ')}", value=float(df[feat].median()))

            if st.button("Predict Approval Status", type="primary"):
                # Prepare data for prediction
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                st.markdown("---")
                if prediction == 1:
                    st.success(f"### 🎉 Result: Approved!")
                    st.write(f"Confidence Score: {probability*100:.1f}%")
                else:
                    st.error(f"### ❌ Result: Rejected")
                    st.write(f"Approval Probability: {probability*100:.1f}%")

    # --- TAB 3: DATA ---
    with tab_data:
        st.subheader("Dataset Exploration")
        st.dataframe(df_filtered, use_container_width=True)
        st.write("### Statistics Summary")
        st.write(df_filtered.describe())

else:
    st.error("⚠️ **File Not Found:** Please ensure 'loan_approval_data.csv' is in the same folder.")