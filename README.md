# 🚀 Language Proficiency Predictor (ML + Streamlit)

## 📌 Overview

This project is an **end-to-end Machine Learning web application** that predicts a user's **language proficiency score** based on their daily practice habits and other features.

The app is built using:

* 🧠 Machine Learning (Scikit-learn)
* 🌐 Streamlit (for UI)
* 📊 Plotly (for visualization)
* 🐳 Docker (for containerization)

---

## 🎯 Features

* 📥 User input-based prediction
* 📊 Interactive data visualization
* 🤖 Trained ML model for predictions
* 🌐 Deployed web application
* 🐳 Docker support for portability

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas, NumPy
* Plotly
* Joblib
* Docker

---

## 📂 Project Structure

```
project/
│── app.py               # Main Streamlit app
│── model.pkl           # Trained ML model
│── requirements.txt    # Dependencies
│── Dockerfile          # Container setup
│── README.md           # Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### Build Image

```bash
docker build -t ml-streamlit-app .
```

### Run Container

```bash
docker run -p 8501:8501 ml-streamlit-app
```

---

## 🌍 Deployment

This app can be deployed using:

* Streamlit Community Cloud
* Docker + Cloud Platforms (AWS / GCP / Azure)

---

## 📊 What the Model Predicts

The model predicts:
👉 **Overall Language Proficiency Score**

Based on:

* Daily practice hours
* Language type
* Learning patterns

---

## ⚠️ Notes

* Ensure all dependencies are listed in `requirements.txt`
* Model file (`model.pkl`) must be present in the root directory
* Remove unsupported features (like `trendline='ols'`) for cloud deployment

---

## 📈 Future Improvements

* Add more features to dataset
* Improve model accuracy
* Add user authentication
* Store predictions in database

---

## 👨‍💻 Author

**Manish (AI/ML Engineer Student)**
3rd Year CSE (AI & ML)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
