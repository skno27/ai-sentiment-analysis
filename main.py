import streamlit as st
from text_analysis import analyze
import os
import time
from validate import check_profanity

st.markdown(
    """
    <style>
    /* General mobile styles */
    @media only screen and (max-width: 768px) {
        /* Adjust page padding and margins */
        .main {
            padding: 0rem 1rem;
        }

        /* Resize header text */
        h1, h2, h3 {
            font-size: 1.5rem;
        }

        /* Make text input boxes and buttons more mobile-friendly */
        .stTextInput > div {
            padding: 0.5rem;
        }

        .stButton > button {
            width: 100%; /* Full-width buttons */
            padding: 1rem;
            font-size: 1rem;
        }

        /* Adjust radio buttons for better spacing */
        .stRadio > div {
            padding: 0.5rem;
        }
    }

    /* General desktop styles (leave as default) */
    @media only screen and (min-width: 768px) {
        .stButton > button {
            width: auto; /* Normal button width */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "last_submission_time" not in st.session_state:
    st.session_state.last_submission_time = 0

pos_folder = "./aclImdb/train/pos"
neg_folder = "./aclImdb/train/neg"

st.title("AI Sentiment Analysis")

st.text(
    "Please show your support for this project by leaving feedback on whether the feedback you received was correct or not."
)

if "comment" not in st.session_state:
    st.session_state.comment = ""
if "sentiment" not in st.session_state:
    st.session_state.sentiment = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


comment = st.text_input("Enter a comment to analyze:")


if st.button("Analyze Comment"):
    if not comment.strip():
        st.write("Please enter a valid comment.")
    elif check_profanity(comment):
        st.warning("This comment doesn't follow quality standards.")
    else:

        result = analyze(comment)
        st.session_state.comment = comment
        st.session_state.sentiment = "Positive" if result[0] > 0.7 else "Negative"
        st.write(f"The sentiment of the comment is: {st.session_state.sentiment}")
        st.session_state.feedback_submitted = False

if st.session_state.sentiment:
    feedback = st.radio(
        "How would you rate this feedback?",
        options=["Good", "Bad"],
        index=0,
        key="feedback_radio",
    )

    if st.button("Submit Feedback"):
        current_time = time.time()
        if current_time - st.session_state.last_submission_time < 10:
            st.warning("Please wait a moment before submitting again.")
        else:
            if not st.session_state.feedback_submitted:
                folder_path = (
                    pos_folder
                    if (feedback == "Good" and st.session_state.sentiment == "Positive")
                    or (feedback == "Bad" and st.session_state.sentiment == "Negative")
                    else neg_folder
                )

                # save feedback/prompt block
                """
                file_name = f"{len(os.listdir(folder_path))}_user.txt"
                file_path = os.path.join(folder_path, file_name)

                with open(file_path, "w") as f:
                    f.write(comment)
                
                """

                st.session_state.feedback_submitted = True
                st.success("Thank you for your feedback")

            else:
                st.warning("Feedback has already been submitted.")

if st.button("Clear"):
    st.session_state.comment = ""
    st.session_state.sentiment = ""
    st.session_state.feedback_submitted = False
    st.rerun()
