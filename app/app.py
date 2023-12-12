import streamlit as st
import openai
import dotenv
import logging
import os


dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# Function to check toxicity using OpenAI's Chat API
def is_toxic(text):
    openai.api_key = os.getenv("OPEN_AI_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Using the latest model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text}"},
        ],
    )

    reply = response["choices"][0]["message"]["content"].strip().lower()

    # Assuming the model's response indicates toxicity or non-toxicity
    return "toxic" in reply


# Streamlit app
def main():
    st.title("Toxicity Checker")

    # Text input
    user_input = st.text_area("Enter text to check for toxicity")

    if user_input:
        # Check for toxicity
        toxic = is_toxic(user_input)

        # Display result
        if toxic:
            st.markdown("<h2 style='color: red;'>Toxic</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>Non-toxic</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
