import streamlit as st

MATRIX_STYLE = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');
    
    /* Main App Background - Pure Black */
    .stApp {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    /* Main Content Container */
    .main .block-container {
        background-color: #000000;
        padding: 2rem;
        max-width: 1200px;
    }
    
    /* Headers - Matrix Green */
    h1, h2, h3, h4, h5, h6 {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
        font-weight: 700 !important;
    }
    
    h1 {
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 0 10px #00ff00;
    }
    
    /* Text and Labels */
    .stMarkdown, .stText, p, div, span, label {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #001100 !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    .stTextInput > label,
    .stTextArea > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #00ff00 !important;
        font-family: 'Courier Prime', monospace !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #001100;
        color: #00ff00;
        border: 2px solid #00ff00;
        font-family: 'Courier Prime', monospace;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00ff00;
        color: #000000;
    }
    
    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background-color: #00ff00;
        color: #000000;
        border: 2px solid #00ff00;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #00cc00;
        border-color: #00cc00;
    }
    
    /* Secondary Button */
    div.stButton > button[kind="secondary"] {
        background-color: #003300;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background-color: #00ff00;
    }
    
    /* Containers and Borders */
    .stContainer, div[data-testid="stContainer"] {
        border: 1px solid #00ff00;
        background-color: #001100;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #00ff00;
    }
    
    /* Sidebar */
    .css-1d391kg, .stSidebar {
        background-color: #000000 !important;
        border-right: 2px solid #00ff00;
    }
    
    /* Fix sidebar content */
    .stSidebar > div {
        background-color: #000000 !important;
    }
    
    /* Fix top header area */
    .stApp > header {
        background-color: #000000 !important;
    }
    
    /* Fix main container top area */
    .main > div {
        background-color: #000000 !important;
    }
    
    /* Fix any remaining white areas */
    div[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    div[data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* Fix selectbox dropdown */
    .stSelectbox > div > div > div {
        background-color: #001100 !important;
        color: #00ff00 !important;
    }
    
    /* Fix slider components */
    .stSlider > div > div > div {
        color: #00ff00 !important;
    }
    
    /* Fix checkbox */
    .stCheckbox > label {
        color: #00ff00 !important;
    }
    
    /* Fix any remaining white backgrounds */
    .element-container, .stMarkdown, .stText {
        background-color: transparent !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #001100;
        border: 1px solid #00ff00;
        color: #00ff00;
    }
    
    .stError {
        background-color: #110000;
        border: 1px solid #ff0000;
        color: #ff0000;
    }
    
    .stWarning {
        background-color: #111100;
        border: 1px solid #ffff00;
        color: #ffff00;
    }
    
    .stInfo {
        background-color: #001111;
        border: 1px solid #00ffff;
        color: #00ffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
        border-bottom: 2px solid #00ff00;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #001100;
        border-bottom: 2px solid #00ff00;
    }
    
    /* JSON Display */
    .stJson {
        background-color: #001100;
        border: 1px solid #00ff00;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #001100;
        border: 1px solid #00ff00;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] > div {
        color: #00ff00;
        font-family: 'Courier Prime', monospace;
    }
    
    /* Remove default Streamlit styling */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > .main > .block-container {
        padding-top: 1rem;
    }
    
    </style>
"""

def apply_matrix_style():
    st.markdown(MATRIX_STYLE, unsafe_allow_html=True)
