import streamlit as st
import openai
import json
import fsspec
import datetime
import dotenv

dotenv.load_dotenv()


# Function to log data to S3
def log_to_s3(bucket_name, data):
    fs = fsspec.filesystem("s3", anon=False)  # Update with AWS credentials if needed

    # Create a unique filename based on the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{bucket_name}/openai_logs/{timestamp}.json"

    # Writing the data to S3
    with fs.open(filename, "w") as f:
        json.dump(data, f)


# Function to check toxicity and log to S3
def is_toxic(text, bucket_name):
    openai.api_key = os.getenv("OPEN_AI_KEY")
    esponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Using the latest model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", 
             "content": ("Please respond with yes or no. "
                         f"Is this comment toxic? {text}")},
        ],
    )
    reply = response["choices"][0]["message"]["content"].strip().lower()
    log_data = {"request": text, "response": reply}
    log_to_s3(bucket_name, log_data)
    return "yes" in reply


# Streamlit app
def main():
    st.title("Toxicity Checker")

    # AWS S3 bucket name
    bucket_name = "your-bucket-name"  # Replace with your S3 bucket name

    # Text input
    user_input = st.text_area("Enter text to check for toxicity")

    if user_input:
        # Check for toxicity and log to S3
        toxic = is_toxic(user_input, bucket_name)

        # Display result
        if toxic:
            st.markdown("<h2 style='color: red;'>Toxic</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>Non-toxic</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
