import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# 1. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    dataset = pd.read_csv(r"C:\Users\satya\Gen Ai_or_AgenticAi\Projects\ClassificationRegression\logit classification.csv")
    return dataset

dataset = load_data()

st.title("🧑‍💻 Purchase Prediction App")
st.write("This app predicts whether a user will purchase based on **Age** and **Estimated Salary** using Random Forest.")

st.subheader("📊 Dataset Preview")
st.dataframe(dataset.head())

# -------------------------------
# 2. Train Model
# -------------------------------
X = dataset.iloc[:, [2, 3]].values   # Age, EstimatedSalary
y = dataset.iloc[:, -1].values       # Purchased

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfc = RandomForestClassifier(
    n_estimators=40,
    criterion='gini',
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=0
)
rfc.fit(X_train, y_train)

# -------------------------------
# 3. Model Evaluation
# -------------------------------
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write("**Confusion Matrix:**")
st.write(cm)

# -------------------------------
# 4. User Input for Prediction
# -------------------------------
st.subheader("🔮 Try Your Own Prediction")

age = st.slider("Select Age", 18, 60, 25)
salary = st.slider("Select Estimated Salary", 15000, 150000, 50000)

user_input = sc.transform([[age, salary]])
prediction = rfc.predict(user_input)[0]

if prediction == 1:
    st.success(f"✅ The model predicts: PURCHASED")
else:
    st.error(f"❌ The model predicts: NOT PURCHASED")

