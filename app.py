import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("placement.csv")

X = data[['cgpa', 'internships', 'skills']]
y = data['placed']

model = LogisticRegression()
model.fit(X, y)

st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

st.title("🎓 AI Student Placement Predictor")
st.write("Predict placement chances using Machine Learning")

st.write("---")

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
internships = st.number_input("Number of Internships", min_value=0, max_value=5)
skills = st.number_input("Number of Skills", min_value=0, max_value=10)

st.write("---")

if st.button("Predict Placement"):
    prediction = model.predict([[cgpa, internships, skills]])
    if prediction[0] == 1:
        st.success("High Chance of Placement")
    else:
        st.error("Low Chance of Placement")

st.write("---")
st.write("Powered by AI & Google Technologies")