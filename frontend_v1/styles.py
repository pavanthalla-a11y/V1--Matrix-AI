
import streamlit as st

def apply_matrix_style():
    style = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

            :root {
                --primary-color: #89b4f8; /* A nice blue from Gemini */
                --background-color: #f0f4f9;
                --card-background-color: #ffffff;
                --text-color: #3c4043;
                --subtle-text-color: #5f6368;
                --border-color: #dfe1e5;
                --font-family: 'Google Sans', sans-serif;
            }

            body {
                font-family: var(--font-family);
                color: var(--text-color);
                background-color: var(--background-color);
            }

            .stApp {
                background-color: var(--background-color);
            }

            h1, h2, h3, h4, h5, h6 {
                font-family: var(--font-family);
                font-weight: 700;
                color: var(--text-color);
            }

            .stButton>button {
                border-radius: 8px;
                padding: 10px 20px;
                font-family: var(--font-family);
                font-weight: 500;
                background-color: var(--primary-color);
                color: white;
                border: none;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: all 0.2s ease-in-out;
            }

            .stButton>button:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                transform: translateY(-2px);
            }

            .step-container {
                background-color: var(--card-background-color);
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid var(--border-color);
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            .stTextInput>div>div>input, .stTextArea>div>div>textarea {
                border-radius: 8px;
                border: 1px solid var(--border-color);
                background-color: var(--background-color);
                padding: 10px;
                font-family: var(--font-family);
            }

            .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 2px rgba(137, 180, 248, 0.5);
            }

            .stSpinner>div>div {
                border-top-color: var(--primary-color);
            }
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
