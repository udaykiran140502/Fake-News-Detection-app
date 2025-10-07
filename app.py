import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to check weather it is Fake or Real")

new_input = st.text_area("News Article:","")

if st.button("Check News"):
    if new_input.strip():
        transform_input= vectorizer.transform([new_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("This news is real")
        else:
            st.error("this is news is fake")
    else:
        st.warning("Please enter some text")