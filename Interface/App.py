import streamlit as st
import pandas as pd
import numpy as np
import joblib
from transformers import XLNetTokenizer, XLNetModel
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from huggingface_hub import hf_hub_download

# Setting up the page configuration
st.set_page_config(page_title="ESApp - Automatic Story Point Estimator", layout="wide", page_icon="🎯")

def fibonacci_sequence(n):
    fib_sequence = [0, 1]
    while fib_sequence[-1] < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

def next_fibonacci(value):
    fib_sequence = fibonacci_sequence(value)
    for fib in fib_sequence:
        if fib >= value:
            return fib
    return None

# Function to load class accuracy file
def load_class_accuracy(lang):
    if lang == "en":
        with open('model/en/class_accuracy_rf.json', 'r') as f:
            accuracy_dict = json.load(f)
        return accuracy_dict
    if lang == "pt":
        with open('model/pt/class_accuracy_pt.json', 'r') as f:
            accuracy_dict = json.load(f)
        return accuracy_dict 

# Function to load XLNet model and tokenizer
def load_xlnet_model():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')
    return tokenizer, model

# Function to generate embeddings with XLNet
def generate_xlnet_embeddings(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Estimation function using Random Forest (after converting to embeddings)
def estimate_story_point_rf(lang, text, rf_model, tokenizer, xlnet_model, max_sequence_length=512):
    embeddings = generate_xlnet_embeddings(text, tokenizer, xlnet_model, max_length=max_sequence_length)
    accuracy_dict = load_class_accuracy(lang)
    prediction = rf_model.predict(embeddings)
    estimated_fib = next_fibonacci(prediction[0])
    accuracy = accuracy_dict.get(str(estimated_fib), 0)
    return estimated_fib, accuracy

# Function to estimate story points using deep learning
def estimate_story_point_deep(text, model, tokenizer, max_sequence_length):
    text = [text]
    encSequences_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(encSequences_test, maxlen=max_sequence_length, padding='post')
    model.compile(optimizer="adam", loss="mse")

    prediction = model.predict(x_test, batch_size=None, verbose=0, steps=None)
    estimate = prediction[0][0]
    return estimate

# Loading the model and tokenizer
model = ""
df = pd.read_csv('reqTxt.csv', header=None)
df_require = df.iloc[:, :]
X = df_require[0]
X = np.array(X)

MAX_LEN = 23313
tokenizer = Tokenizer(num_words=MAX_LEN, char_level=False, lower=False)
tokenizer.fit_on_texts(X)
MAX_SEQUENCE_LENGTH = 100

st.sidebar.title("Select Estimation Model")
option_model = st.sidebar.selectbox("Choose the story point estimation method:", ("Numeric - en"))

lang = ""

#if option_model == "Fibonacci - en":
#    model = joblib.load('model/en/rf.joblib')
#    lang = "en"

#if option_model == "Fibonacci - pt":
#    model = joblib.load('model/pt/rf_pt.joblib')
#    lang = "pt"

if option_model == "Numeric - en":
    # Baixa o modelo do Hugging Face
    model_path = hf_hub_download(repo_id="DanielOS/ESApp-en", filename="deep_learning_regressao.pkl")

    # Carrega o modelo com joblib
    model = joblib.load(model_path)
    print( "Este é o tipo do modelo: ",type(model))  # Deve exibir o tipo correto, ex: <class 'sklearn.ensemble._forest.RandomForestRegressor'>
    model.compile(optimizer="adam", loss="mse")
    lang = "en"

#if option_model == "Numeric - pt":
 #   model = joblib.load('model/pt/deep_learning_pt.pkl')
  #  lang = "pt"

st.title('🎯 ESApp - Automatic Story Point Estimator')
st.markdown("""
Welcome to ESApp! This tool helps you automatically estimate story points using a machine learning model.
You can input a single user story description or upload a CSV file with multiple stories.
""")

if 'estimate' not in st.session_state:
    st.session_state.estimate = None
if 'correction' not in st.session_state:
    st.session_state.correction = 0

if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None

st.sidebar.title("Input Options")
option = st.sidebar.radio("Choose the input method:", ("Manual Input", "Upload CSV File"))

if option == "Manual Input":
    st.subheader('Input Single User Story')
    

    user_input = st.text_area("Enter the story description:")

    if st.button("Estimate Story Point"):
        if user_input:
            if lang == "":
                lang = "en"
            if "Numeric" in option_model:
                with st.spinner('Estimating story point...'):
                    st.session_state.estimate = estimate_story_point_deep(user_input, model, tokenizer, MAX_SEQUENCE_LENGTH)
                st.success(f"Suggested story point: {st.session_state.estimate:.1f}")
                st.info(f"This model has an average error of 3.77 for assertiveness" if lang == "en" else "Este modelo possui um erro médio de 3.91 na assertividade")

            if "Fibonacci" in option_model:
                with st.spinner('Estimating story point...'):
                    tokenizer, xlnet_model = load_xlnet_model()
                    st.session_state.estimate, st.session_state.accuracy = estimate_story_point_rf(lang, user_input, model, tokenizer, xlnet_model)
                st.success(f"Suggested story point: {st.session_state.estimate}")
                st.info(f"For this estimate, this model has an assertiveness of: {st.session_state.accuracy:.1f}%")

            st.session_state.correction = st.number_input("If the estimate is incorrect, enter the correct value here:", min_value=1)

    if st.button("Save Correction"):
        if st.session_state.correction != st.session_state.estimate:
            with open('corrections.csv', 'a') as f:
                f.write(f"{user_input},{st.session_state.estimate},{st.session_state.correction}\n")
            st.success("Correction saved successfully!")
        else:
            st.warning("The correction is the same as the original estimate. No save was needed.")
elif option == "Upload CSV File":
    st.subheader('Upload CSV File with User Stories')
    st.markdown("""
    **CSV File Format:**
    - It must contain at least three columns: ID, Title, and Story.
    """)
    example_csv = pd.DataFrame({
        'ID': [1, 2],
        'Title': ['System Login', 'User Management'],
        'Story': [
            'As a user, I want to log in to the system to access my personal information.',
            'As an administrator, I want to manage users to keep the system organized and secure.'
        ]
    })
    csv = example_csv.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Example CSV",
        data=csv,
        file_name='example.csv',
        mime='text/csv'
    )
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if st.button("Estimate Points by Stories"):
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        if len(df_uploaded.columns) < 3:
            st.error("The CSV file must have at least three columns (ID, Title, Story).")
        else:
            with st.spinner('Estimating points by stories...'):
                stories = df_uploaded.iloc[:, 2].tolist()
                if "Numeric" in option_model:
                    estimates = [estimate_story_point_deep(story, model, tokenizer, MAX_SEQUENCE_LENGTH) for story in stories]
                    if lang == "en":
                        st.info(f"This model has an average error of 3.77 in accuracy.")
                    if lang == "pt":
                        st.info(f"This model has an average error of 3.91 in accuracy.")
                    
                    st.session_state.df_uploaded = df_uploaded.copy()
                    st.session_state.df_uploaded['estimate'] = estimates
                if "Fibonacci" in option_model:
                    tokenizer, model_xlnet = load_xlnet_model()
                    results = [estimate_points_by_story_random_forest(lang, story, model, tokenizer, model_xlnet) for story in stories]
                    estimates, accuracies = zip(*results)
                    print("estimate:", estimates)

                    # st.info(f"For this estimate, this model has: {st.session_state.accuracy:.1f}% accuracy")
                   
                    st.session_state.df_uploaded = df_uploaded.copy()
                    st.session_state.df_uploaded['estimate'] = estimates
                    st.session_state.df_uploaded['accuracy'] = accuracies
                   
                    # Initializing corrections with the initial estimates
                    st.session_state.corrections = list(estimates).copy()
                st.success("Estimates generated successfully!")
                st.write(st.session_state.df_uploaded)

                # Allow downloading the file with estimates
                csv = st.session_state.df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV with Estimates", data=csv, file_name="estimates.csv", mime='text/csv')

                # Option for manual correction of estimates
            st.subheader('Manual Correction of Estimates')
            corrections = []
        corrections_estimates = []
        for i in range(len(st.session_state.df_uploaded)):
            x = st.number_input(f"Correction for ID {st.session_state.df_uploaded.iloc[i, 0]} (Original Estimate: {st.session_state.df_uploaded.iloc[i, -1]}):", min_value=0, value=int(st.session_state.corrections[i]), key=f"correction_{i}")
            st.session_state.corrections[i] = x
        # Updating the DataFrame with the corrections
        st.session_state.df_uploaded['Correction'] = st.session_state.corrections

        # Allow downloading the file with estimates and corrections
        corrected_csv = st.session_state.df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Estimates and Corrections", data=corrected_csv, file_name="corrected_estimates.csv", mime='text/csv')
    
    else:
        st.warning("Please upload a CSV file containing the stories.")

# Add a link for documentation or additional help
st.sidebar.markdown("For more information, visit the [documentation](https://link_to_thesis).")

# Add help icon with tooltip about the CSV file format
if option == "Upload CSV File":
    st.sidebar.markdown("""
        <style>
        .info-icon {
            display: inline-flex;
            align-items: center;
        }
        .info-icon img {
            margin-right: 5px;
        }
        </style>
        <div class="info-icon">
            <img src="https://img.icons8.com/ios-filled/50/000000/info.png" width="20" height="20">
            <span>CSV File Format</span>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.info("""
    The CSV file must contain at least three columns: ID, Title, and Story.
    """)
st.sidebar.markdown("""
---
**Note:** The estimate provided is based on a machine learning model to assist the development team. Further analysis by the team is recommended for validation and necessary adjustments.
""")
