# 🏦 FinScore: ML-Driven Credit Decision Engine

An end-to-end Machine Learning powered loan approval analytics and credit risk decision platform. **FinScore** ingests financial applicant data, cleans missing values dynamically through automated imputation, trains a Logistic Regression risk classifier, and serves real-time loan approval predictions with probability confidence scores via an interactive Streamlit dashboard.

---

## 🌟 Key Features

* **📊 Interactive Analytics Dashboard**: Live metrics (*Total Applications*, *Avg Income*, *Approval Rate %*, *Avg Credit Score*), Plotly Donut charts for approval distribution, and Income vs. Credit Score scatter plots.
* **🚀 Real-Time Loan Eligibility Calculator**: Instant ML-based credit risk assessment with probability confidence scores.
* **📋 Dual-Mode Raw Data Management**:
  * **User-Specific View**: Instantly inspect records created during the current session or search by specific `Applicant_ID`.
  * **Full Dataset View**: Browse, filter, and sort all 1,000+ dataset records.
* **📄 One-Click Master PDF & CSV Export**: Generate formatted, multi-page PDF credit reports (using ReportLab) or export CSV data on demand.
* **🛡️ Fault-Tolerant & Reactive**: Graceful NaN handling, automated imputations, session state synchronization, and zero runtime crashes.

---

## 🏗️ System Architecture & Workflow Flowcharts

### 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    A[loan_approval_data.csv] -->|Pandas Ingestion| B[Data Preprocessing Layer]
    B -->|Mean & Mode Imputation| C[Cleaned Dataset]
    
    C -->|Feature Extraction| D[Logistic Regression Model]
    C -->|Session State Sync| E[Streamlit Reactive UI]
    
    D -->|Predict Probability| F[Real-Time Loan Predictor]
    
    E --> G[📊 Analytics Dashboard]
    E --> H[🚀 Loan Eligibility Engine]
    E --> I[📋 Dual-Mode Raw Data Viewer]
    
    H -->|Append Record| C
    H -->|Save Disk| A
    
    I -->|ReportLab Engine| J[📄 Formatted PDF Export]
    I -->|UTF-8 Encoding| K[📊 CSV Data Export]
```

---

### 2. User Journey & Data Pipeline Flowchart

```mermaid
sequenceDiagram
    autonumber
    actor User as Applicant / Loan Officer
    participant UI as Streamlit Web Interface
    participant State as Session State (st.session_state)
    participant Model as Logistic Regression ML Model
    participant Disk as CSV Storage (loan_approval_data.csv)
    participant PDF as ReportLab PDF Engine

    User->>UI: Enter Applicant Financials (Income, Credit Score, Age, Loans)
    UI->>Model: Pass Feature Input Array
    Model-->>UI: Return Prediction (Approved/Rejected) + Confidence Score %
    UI->>State: Append New Applicant Record to Session Dataset
    UI->>Disk: Persist Updated Data to Disk
    UI->>UI: Trigger st.rerun() for Instant Cross-Tab Sync
    UI-->>User: Display Immediate Prediction Result Banner
    User->>UI: Switch to Raw Data Tab
    UI-->>User: Display Updated Table (User-Specific or Full View)
    User->>UI: Click "Download PDF Report"
    UI->>PDF: Compile Table & Metrics into Landscape PDF
    PDF-->>User: Download Formatted Master PDF Document
```

---

## 🛠️ Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Core application logic |
| **Web Framework** | Streamlit | Interactive web application frontend & state management |
| **Data Processing** | Pandas, NumPy, scikit-learn (`SimpleImputer`) | Data cleaning, mean/mode imputation & manipulation |
| **Machine Learning** | scikit-learn (`LogisticRegression`) | Binary classification for credit risk assessment |
| **Data Visualization**| Plotly Express | Interactive charts (Donut charts, Scatter plots) |
| **Document Export** | ReportLab | Programmatic PDF generation with styled tables |

---

## ⚡ How to Run the Project

### 1. Prerequisites
Ensure you have **Python 3.8+** installed:
```bash
python --version
```

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/sawantyash07/-FinScore-ML-Driven-Credit-Decision-Engine.git
cd -FinScore-ML-Driven-Credit-Decision-Engine
pip install -r requirements.txt plotly reportlab
```

### 3. Launch Web Application
```bash
streamlit run app.py
```
> The application will open automatically in your browser at `http://localhost:8501` (or `http://localhost:8502`).



## 📂 Project Structure

```
FinScore-ML-Driven-Credit-Decision-Engine/
├── app.py                            # Main Streamlit Application (Dashboard, Predictor, Raw Data)
├── app.pu.py                         # Alternate Matplotlib/Seaborn visualization dashboard
├── loan_approval_data.csv            # Primary dataset (1,000+ applicant records)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation & interview guide
└── complete code/
    └── credit_wise.ipynb             # Jupyter Notebook for EDA & model exploration
```

---

## 📜 License

Distributed under the MIT License. Feel free to use, modify, and build upon this project for learning and interview preparation!
