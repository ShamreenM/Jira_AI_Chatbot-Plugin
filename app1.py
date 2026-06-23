import streamlit as st
from JiraAIChatbot4 import retrieve_docs
from dotenv import load_dotenv
import os

load_dotenv("key.env")

apiKey = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="JIRA AI Chatbot")

st.title("🤖 JIRA AI Chatbot")

user_input = st.text_input(
    "Ask your JIRA question:"
)

if st.button("Submit"):

    if user_input:

        with st.spinner(
            "🔍 Searching JIRA issues..."
        ):

            result = retrieve_docs(
                user_input,
                apiKey
            )

        st.markdown("### Response")

        st.text(result)

    else:
        st.warning(
            "Please enter a question."
        )