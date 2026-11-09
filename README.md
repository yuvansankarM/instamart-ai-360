
# 🛒 Instamart AI 360
### AI-Powered Customer Intelligence & Autonomous Support Platform

An end-to-end data science and machine learning project inspired by quick-commerce platforms.

Instamart AI 360 combines customer segmentation, churn prediction, support ticket classification, and an interactive analytics dashboard.

## 🚀 Project Features

### 1. Customer Intelligence
- RFM (Recency, Frequency, Monetary) analysis
- Customer segmentation
- Customer behaviour analytics

### 2. Churn Prediction
- Random Forest classification
- Customer churn probability
- High, Medium, and Low risk categorization
- Customer risk report

### 3. Support Ticket Intelligence
- TF-IDF text vectorization
- Logistic Regression classification
- Automated ticket category prediction
- Ticket category analytics

### 4. Interactive Dashboard
- Streamlit application
- Dark premium interface
- Customer analytics
- Churn analytics
- Ticket intelligence

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Models | Random Forest, Logistic Regression |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Development | VS Code, Jupyter Notebook |

## 📁 Project Structure

```text
instamart-ai-360/
│
├── data/
│   ├── synthetic/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_customer_intelligence.ipynb
│   ├── 03_churn_prediction.ipynb
│   └── 04_ticket_intelligence.ipynb
│
├── src/
│
├── models/
│   └── ticket_classifier.pkl
│
├── dashboard/
│   └── app.py
│
├── documents/
├── tests/
├── config/
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd instamart-ai-360
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

## 🧠 Machine Learning Models

### Churn Prediction
Algorithm: Random Forest Classifier

Purpose: Estimate the probability that a customer will stop purchasing during the defined future observation window.

### Ticket Classification
Algorithm: TF-IDF + Logistic Regression

Purpose: Predict the category of a customer support message.

## 📊 Project Outcomes

- Built an RFM-based customer segmentation system.
- Developed a baseline customer churn prediction model.
- Created an NLP-based support ticket classifier.
- Built an interactive Streamlit analytics dashboard.
- Generated customer risk reports.

## 🔮 Future Enhancements

- Retrieval-Augmented Generation (RAG) support assistant
- Sentiment prediction model
- Duplicate ticket detection
- Automated ticket routing
- Live data integration
- Model monitoring and evaluation

## ⚠️ Disclaimer

This is an educational portfolio project using synthetic customer, order, and support-ticket data.

It is not an official Swiggy or Instamart application and does not use real customer information.

## 👨‍💻 Author

**Your Name**

Data Science & Machine Learning Project
