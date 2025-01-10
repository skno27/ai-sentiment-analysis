from profanity import profanity
import streamlit as st

# validates the input prompt


def check_profanity(prompt):
    return profanity.contains_profanity(prompt)
