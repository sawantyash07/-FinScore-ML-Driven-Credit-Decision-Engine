import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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

# 2. PDF Generator Function
def generate_pdf_report(df_to_export):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=15,
        leftMargin=15,
        topMargin=20,
        bottomMargin=20
    )
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12
    )
    cell_style = ParagraphStyle('Cell', fontSize=5.5, leading=6.5, textColor=colors.HexColor('#1E293B'))
    header_style = ParagraphStyle('Header', fontSize=6, leading=7, fontName='Helvetica-Bold', textColor=colors.white)

    elements.append(Paragraph("CreditWise Loan Management System — Master Applicant Report", title_style))
    total_recs = len(df_to_export)
    approved_count = (df_to_export['Loan_Approved'].astype(str).str.upper().isin(['YES', '1'])).sum()
    rate = (approved_count / total_recs * 100) if total_recs > 0 else 0
    date_str = pd.Timestamp.now().strftime('%d %B %Y, %H:%M:%S')

    elements.append(Paragraph(f"<b>Generated Date:</b> {date_str} &nbsp;|&nbsp; <b>Total Applicants:</b> {total_recs:,} &nbsp;|&nbsp; <b>Approved:</b> {approved_count:,} ({rate:.1f}%)", sub_style))

    cols = list(df_to_export.columns)
    table_data = [[Paragraph(str(col).replace('_', ' '), header_style) for col in cols]]

    for _, row in df_to_export.iterrows():
        row_cells = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                val_str = "-"
            elif isinstance(val, (float, np.floating)):
                if col in ['Applicant_Income', 'Coapplicant_Income', 'Savings', 'Collateral_Value', 'Loan_Amount']:
                    val_str = f"${val:,.0f}"
                elif val.is_integer():
                    val_str = f"{int(val)}"
                else:
                    val_str = f"{val:.2f}"
            elif isinstance(val, (int, np.integer)):
                val_str = f"{val}"
            else:
                val_str = str(val)
            row_cells.append(Paragraph(val_str, cell_style))
        table_data.append(row_cells)

    pdf_table = Table(table_data, repeatRows=1)
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))

    elements.append(pdf_table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# 3. Data Loading & Session State Initialization
def load_and_clean_data():
    df = pd.read_csv('loan_approval_data.csv')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    return df

if 'df' not in st.session_state:
    try:
        st.session_state.df = load_and_clean_data()
    except Exception as e:
        st.session_state.df = None

df = st.session_state.df

if df is not None:
    # Train Logistic Regression Model on current dataframe
    feature_list = ['Applicant_Income', 'Credit_Score', 'Age', 'Existing_Loans']
    available_features = [f for f in feature_list if f in df.columns]
    
    X = df[available_features]
    y = df['Loan_Approved'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # 4. Sidebar Controls & Filtering
    st.sidebar.title("🎮 Dashboard Controls")
    
    if 'Employment_Status' in df.columns:
        options = sorted(df['Employment_Status'].astype(str).unique().tolist())
        emp_status = st.sidebar.multiselect("Filter by Employment", options, default=options)
        df_filtered = df[df['Employment_Status'].isin(emp_status)]
    else:
        df_filtered = df

    with st.sidebar.expander("⚙️ Data Management"):
        st.write(f"Total Rows in Dataset: **{len(df)}**")
        if st.button("Reload Original Dataset from CSV"):
            st.session_state.df = load_and_clean_data()
            if 'last_prediction' in st.session_state:
                del st.session_state.last_prediction
            st.rerun()

    # 5. Main Tabs
    tab_dashboard, tab_predictor, tab_data = st.tabs(["📊 Analytics Dashboard", "🚀 Loan Predictor", "📋 Raw Data"])

    # --- TAB 1: DASHBOARD ---
    with tab_dashboard:
        st.title("💰 Loan Approval Analytics")
        if df_filtered.empty:
            st.warning("⚠️ No data matches the selected filter criteria. Please select at least one Employment Status filter.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Applications", f"{len(df_filtered):,}")
            with col2:
                avg_inc = df_filtered['Applicant_Income'].mean()
                st.metric("Avg. Income", f"${avg_inc:,.0f}" if pd.notnull(avg_inc) else "N/A")
            with col3:
                mapped = df_filtered['Loan_Approved'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
                rate = mapped.mean() * 100 if pd.notnull(mapped.mean()) else 0.0
                st.metric("Approval Rate", f"{rate:.1f}%")
            with col4:
                avg_credit = df_filtered['Credit_Score'].mean()
                st.metric("Avg. Credit Score", f"{int(avg_credit)}" if pd.notnull(avg_credit) else "N/A")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df_filtered, names='Loan_Approved', hole=0.5, title="Approval Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                fig_scatter = px.scatter(df_filtered, x='Applicant_Income', y='Credit_Score', color='Loan_Approved', title="Income vs Credit Score")
                st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 2: PREDICTOR ---
    with tab_predictor:
        st.title("🚀 Real-Time Loan Eligibility")
        st.markdown("Enter applicant details below to predict loan approval.")
        
        # Display results of latest prediction if available
        if 'last_prediction' in st.session_state:
            res = st.session_state.last_prediction
            st.markdown("---")
            if res['status'] == 'Yes':
                st.success(f"### 🎉 Result: Approved! (Confidence: {res['probability']*100:.1f}%)")
            else:
                st.error(f"### ❌ Result: Rejected (Approval Probability: {res['probability']*100:.1f}%)")

            st.info(f"✅ **Raw Dataset Updated!** Applicant ID `#{int(res['new_id'])}` has been appended to the Raw Data table.")

        with st.container():
            col_a, col_b = st.columns(2)
            user_input = {}
            
            # Dynamically create inputs based on the features the model uses
            for i, feat in enumerate(available_features):
                with col_a if i % 2 == 0 else col_b:
                    median_val = float(df[feat].median()) if pd.notnull(df[feat].median()) else 0.0
                    user_input[feat] = st.number_input(f"Enter {feat.replace('_', ' ')}", value=median_val)

            if st.button("Predict Approval Status", type="primary"):
                # Prepare data for prediction
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                status_str = 'Yes' if prediction == 1 else 'No'

                # Construct new full row matching all df columns
                new_id = float(df['Applicant_ID'].max() + 1) if ('Applicant_ID' in df and len(df) > 0) else 1.0
                
                new_row = {}
                for col in df.columns:
                    if col == 'Applicant_ID':
                        new_row[col] = new_id
                    elif col in user_input:
                        new_row[col] = user_input[col]
                    elif col == 'Loan_Approved':
                        new_row[col] = status_str
                    else:
                        # Fill remaining fields with median (if numeric) or mode (if categorical)
                        if df[col].dtype in ['float64', 'int64']:
                            new_row[col] = float(df[col].median()) if pd.notnull(df[col].median()) else 0.0
                        else:
                            mode_vals = df[col].mode()
                            new_row[col] = str(mode_vals[0]) if not mode_vals.empty else 'N/A'

                # Append to session state dataset
                updated_df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state.df = updated_df

                # Track user created applicant IDs in session state
                if 'user_added_ids' not in st.session_state:
                    st.session_state.user_added_ids = []
                st.session_state.user_added_ids.append(new_id)

                # Save back to CSV for persistent storage across server reloads
                updated_df.to_csv('loan_approval_data.csv', index=False)

                # Store result in session state and trigger immediate rerun so all tabs update
                st.session_state.last_prediction = {
                    'status': status_str,
                    'probability': probability,
                    'new_id': new_id
                }
                st.rerun()

    # --- TAB 3: DATA ---
    with tab_data:
        st.subheader("📋 Dataset Exploration & Report Export")
        
        # Display Mode Toggle: User Specific vs Entire CSV
        view_mode = st.radio(
            "Select Data View Mode:",
            ["👤 Display Only User Specific / Created Records", "📊 Display Entire CSV File Data"],
            horizontal=True
        )

        user_ids = st.session_state.get('user_added_ids', [])

        if view_mode == "👤 Display Only User Specific / Created Records":
            if user_ids:
                display_df = df[df['Applicant_ID'].isin(user_ids)].sort_values(by='Applicant_ID', ascending=False)
                st.info(f"Showing **{len(display_df)}** record(s) created by you during this session.")
            else:
                st.warning("⚠️ No user-created records found for this session yet. Go to **🚀 Loan Predictor** tab to submit a new loan application!")
                # Search fallback if user wants to look up a specific Applicant ID
                search_id = st.number_input("Or search specific Applicant ID:", min_value=1, step=1, value=1)
                display_df = df[df['Applicant_ID'] == search_id]
                if display_df.empty:
                    st.info(f"No record found for Applicant ID #{search_id}.")
        else:
            # Display Entire CSV File Data
            sort_order = st.selectbox("Sort Table By:", ["Most Recent First (Newest Additions Top)", "Original Order (Ascending ID)"])
            if "Most Recent First" in sort_order:
                display_df = df_filtered.sort_values(by='Applicant_ID', ascending=False)
            else:
                display_df = df_filtered.sort_values(by='Applicant_ID', ascending=True)
            st.write(f"Displaying **{len(display_df)}** records out of **{len(df)}** total applications in the system.")

        if not display_df.empty:
            col_pdf, col_csv = st.columns([1, 1])
            
            with col_pdf:
                pdf_data = generate_pdf_report(display_df)
                st.download_button(
                    label="📄 Download Selected View as PDF",
                    data=pdf_data,
                    file_name=f"Loan_Data_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

            with col_csv:
                csv_data = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Selected View as CSV",
                    data=csv_data,
                    file_name=f"Loan_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            st.dataframe(display_df, use_container_width=True, height=500)
            
            st.write("### 📈 Statistics Summary")
            st.write(display_df.describe())

else:
    st.error("⚠️ **File Not Found:** Please ensure 'loan_approval_data.csv' is in the same folder.")