# Malicious-url-detector
Machine learning-based web application to detect phishing and malicious URLs using Flask and XGBoost.
# 🔐 Malicious URL Detection System

## 📌 Overview
This project is a machine learning-based web application developed using Flask to detect malicious and phishing URLs. It analyzes input URLs and classifies them as safe or suspicious using trained models.

## 🚀 Features
- Detects phishing and malicious URLs
- Real-time URL analysis
- User-friendly web interface
- Machine learning-based prediction

## 🛠️ Technologies Used
- Python
- Flask
- Scikit-learn
- XGBoost
- Pandas & NumPy
- tldextract

## ⚙️ How It Works
1. User enters a URL
2. System extracts features from the URL
3. Machine learning model analyzes the features
4. Output is displayed as **Safe** or **Malicious**

## ▶️ How to Run Locally
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
