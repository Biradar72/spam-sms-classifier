import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam SMS Detector")

st.write("Enter message or upload txt file")

text_input = st.text_area("Enter Message")

if st.button("Predict Text"):
    if text_input.strip() != "":
        
    
        text_vec = vectorizer.transform([text_input])
        
        prediction = model.predict(text_vec)
        
        st.write("Raw Prediction:", prediction)

    if prediction[0] == "spam":
        st.error("SPAM")
    else:
        st.success("NOT SPAM")


uploaded_file = st.file_uploader("Upload TXT File", type=["txt"])

if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")

    file_vec = vectorizer.transform([file_text])
    prediction = model.predict(file_vec)

    st.write("Raw Prediction:", prediction)

    if prediction[0] == "spam":
        st.error("SPAM")
    else:
        st.success("NOT SPAM")
