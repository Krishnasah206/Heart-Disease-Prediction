## 💓 Heart Disease Prediction using Machine Learning

This project uses a machine learning model to predict the likelihood of heart disease based on clinical parameters. The model is trained on the Heart Disease Dataset and deployed with a simple interactive **Streamlit web app**.

---

### 🚀 Features

- Trained using **Random Forest Classifier**
- Interactive **Streamlit** frontend
- Saves and loads model/scaler using `joblib`
- Predicts heart disease from user inputs
- Clean UI with instant feedback

---

### 🧠 Dataset Description

The dataset includes the following features:

| Feature     | Description                                      |
|-------------|--------------------------------------------------|
| age         | Age of the patient                               |
| sex         | Sex (1 = male; 0 = female)                       |
| cp          | Chest pain type (0–3)                            |
| trestbps    | Resting blood pressure                           |
| chol        | Serum cholesterol in mg/dl                       |
| fbs         | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg     | Resting electrocardiographic results (0–2)       |
| thalach     | Maximum heart rate achieved                      |
| exang       | Exercise induced angina (1 = yes; 0 = no)        |
| oldpeak     | ST depression induced by exercise                |
| slope       | Slope of the peak exercise ST segment (0–2)      |
| ca          | Number of major vessels (0–4)                    |
| thal        | Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect) |
| target      | 0 = No heart disease, 1 = Heart disease          |

---

### 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

---

### 🧪 How to Run Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Example <code>requirements.txt</code></summary>

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

</details>

#### 3. Train and Save Model

In `heart_disease_prediction.ipynb`, run all cells to:
- Load data
- Train the Random Forest model
- Save the model and scaler

This will create:
- `heart_disease_model.pkl`
- `scaler.pkl`

#### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 📁 Project Structure

```
├── app.py                        # Streamlit app
├── heart.csv                    # Dataset
├── heart_disease_prediction.ipynb  # Notebook with model training
├── heart_disease_model.pkl      # Trained model
├── scaler.pkl                   # Saved scaler
├── requirements.txt
└── README.md
```

### 📌 To-Do / Improvements

- Add model selection (Logistic Regression, SVM, etc.)
- Improve UI with charts and visual explanations
- Deploy on Streamlit Cloud or Hugging Face Spaces