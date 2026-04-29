# 💳 UPI Fraud Detection using AI/ML

A complete end-to-end Machine Learning project that detects fraudulent UPI (Unified Payments Interface) transactions using a **Random Forest Classifier**, along with an interactive **Streamlit Dashboard** for real-time data visualization and analysis.

---

## 📌 Project Overview

With the rapid growth of digital payments in India, UPI fraud has become a major concern. This project builds a machine learning model to automatically detect whether a transaction is **fraudulent or legitimate** based on transaction features like type, step, and fraud rate. It also includes a fully interactive dashboard built with Streamlit for visualizing fraud patterns.

---

## 🎯 Objectives

- Analyze a large-scale financial transaction dataset
- Perform data cleaning, preprocessing, and exploratory data analysis (EDA)
- Build and evaluate a Random Forest Classification model to detect fraud
- Deploy an interactive Streamlit dashboard for visual fraud analysis

---

## 📂 Dataset

- **Name:** PaySim Financial Fraud Dataset
- **Source:** [Kaggle — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **File:** `PS_20174392719_1491204439457_log.csv`
- **Size:** ~470 MB (too large for GitHub — download from Kaggle)
- **Rows:** 6.3 million+ transactions
- **Key Columns:**

| Column | Description |
|--------|-------------|
| `step` | Time step (1 step = 1 hour) |
| `type` | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Transaction amount |
| `nameOrig` | Origin account |
| `oldbalanceOrg` | Balance before transaction (origin) |
| `newbalanceOrig` | Balance after transaction (origin) |
| `nameDest` | Destination account |
| `oldbalanceDest` | Balance before transaction (destination) |
| `newbalanceDest` | Balance after transaction (destination) |
| `isFraud` | Target label — 1 if fraudulent, 0 if legitimate |
| `isFlaggedFraud` | Flag for large illegal attempts |

---

## 🗂️ Project Structure

```
UPI-Fraud-Detection/
│
├── UPI_Fraud_Detection.ipynb   # Main ML notebook (EDA + Model Training)
├── fraud_dashboard.py          # Streamlit interactive dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

> ⚠️ The dataset CSV is not included due to its large size (~470MB). Download it from Kaggle using the link above and place it in the same project folder.

---

## 🔄 Project Workflow

### 📓 Notebook — `UPI_Fraud_Detection.ipynb`

**Step 1 — Data Loading**
- Load the CSV dataset using Pandas

**Step 2 — Exploratory Data Analysis (EDA)**
- `df.head()`, `df.describe()`, `df.shape()` for initial exploration
- Check for duplicate values → `df.duplicated().sum()`
- Check for missing values → `df.isnull().sum()`
- Count unique values per column → `df.nunique()`

**Step 3 — Data Cleaning**
- Drop high-cardinality/irrelevant columns: `nameOrig`, `nameDest`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`
- Group transactions by type to count fraudulent transactions
- Engineer a new feature: `fraud_rate_by_type` (fraud rate per transaction type)

**Step 4 — Data Visualization**
- Bar chart: Number of fraudulent transactions by transaction type
- Bar chart: Percentage of fraudulent transactions by transaction type
- Line plot: Fraudulent transactions over time (step)
- Confusion matrix heatmap after model evaluation

**Step 5 — Model Building**
- One-hot encode the `type` column using `pd.get_dummies()`
- Split data: 70% train / 30% test (`random_state=42`)
- Train a **Random Forest Classifier**
- Evaluate using Classification Report (Precision, Recall, F1-Score)
- Visualize results with a Confusion Matrix

---

## 📊 Streamlit Dashboard — `fraud_dashboard.py`

An interactive dashboard that provides visual fraud analysis with the following features:

| Feature | Description |
|---------|-------------|
| 📋 Raw Data View | Toggle to show/hide the raw dataset |
| 📊 Fraud by Type (Bar Chart) | Count of fraudulent transactions per type |
| 📈 Fraud Percentage by Type | Fraud rate (%) per transaction type |
| 📉 Amount Distribution | Histogram of transaction amounts (log scale) |
| 🕒 Fraud Over Time | Line chart of fraud count per time step |
| 🔥 Correlation Heatmap | Heatmap of numeric feature correlations |
| 🔍 Filter by Type (Sidebar) | Filter all data by a specific transaction type |
| 🥧 Fraud Distribution (Filtered) | Fraud vs Non-Fraud % for the selected type |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML model (Random Forest), train-test split, metrics |
| Imbalanced-learn | Handling class imbalance |
| Streamlit | Interactive web dashboard |
| Pyforest | Auto-imports common data science libraries |

---

## 🚀 How to Run

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Preetham9638/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Dataset
Download the dataset from Kaggle:
👉 https://www.kaggle.com/datasets/ealaxi/paysim1

Place the CSV file in the same folder as the project files.

### Step 4 — Run the Jupyter Notebook
Open `UPI_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab and run all cells.

### Step 5 — Run the Streamlit Dashboard
```bash
python -m streamlit run fraud_dashboard.py
```
The dashboard will open at `http://localhost:8501` in your browser.

---

## 📈 Model Performance

The Random Forest Classifier is evaluated using:
- **Precision** — How many predicted frauds are actually fraud
- **Recall** — How many actual frauds were correctly detected
- **F1-Score** — Balance between Precision and Recall
- **Confusion Matrix** — Visual breakdown of correct and incorrect predictions

---

## 💡 Key Insights

- Fraud occurs **only in TRANSFER and CASH-OUT** transaction types
- The dataset is highly **imbalanced** — fraudulent transactions are a very small percentage
- **Random Forest** handles this well due to its ensemble nature
- Transaction **step (time)** shows patterns in when fraud is more likely to occur

---

## 🙋‍♂️ Author

Built by **Preetham** as an AI/ML project for fraud detection in digital payments.
Feel free to ⭐ star the repo if you found it useful!

---

## 📜 License

This project is open-source and free to use for educational purposes.
