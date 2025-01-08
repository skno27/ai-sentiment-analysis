import streamlit as st
from text_analysis import analyze

st.title("AI Sentiment Analysis")
comment = st.text_input("Enter a comment to analyze:")


if st.button("Analyze Comment"):
    if not comment.strip():
        st.write("Please enter a valid comment.")
    result = analyze(comment)
    sentiment = "Positive" if result[0] > 0.5 else "Negative"
    st.write(f"The sentiment of the comment is: {sentiment}")
