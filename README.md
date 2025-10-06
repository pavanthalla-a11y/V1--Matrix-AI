# Matrix AI - Synthetic Data Generator

Matrix AI is a high-performance, AI-powered synthetic data generation tool that leverages Google's Gemini Pro and the Synthetic Data Vault (SDV) library. It provides a streamlined, four-step process to describe, design, synthesize, and download realistic, tabular data for AI/ML model training, testing, and data augmentation.

## Features

*   **AI-Powered Schema Design:** Describe your desired dataset in natural language, and Matrix AI's Gemini-powered agent will automatically generate the relational schema and realistic seed data.
*   **High-Performance Synthesis:** Utilizes the Synthetic Data Vault (SDV) library to generate large-scale datasets based on the AI-generated schema. The process is optimized for performance with batch processing and faster synthesizers.
*   **Relational Data Support:** Generate complex, multi-table datasets with primary and foreign key relationships, ensuring referential integrity.
*   **Data Quality and Realism:** The AI agent is designed to produce high-quality, realistic data, including names, addresses, and other common data types.
*   **Interactive Web Interface:** A Streamlit-based web application guides users through the four-step data generation process.
*   **Asynchronous Processing:** Data synthesis runs as a background task, allowing users to monitor progress and receive email notifications upon completion.
*   **Comprehensive Reporting:** After synthesis, the tool provides detailed reports on data distribution and generation metrics.
*   **Easy Download:** Download the complete dataset, including all tables and reports, as a single ZIP file.

## Technologies Used

*   **Backend:**
    *   [FastAPI](https://fastapi.tiangolo.com/): A modern, fast (high-performance) web framework for building APIs with Python.
    *   [Uvicorn](https://www.uvicorn.org/): A lightning-fast ASGI server implementation.
    *   [Vertex AI](https://cloud.google.com/vertex-ai): Google's unified AI platform, used to access the Gemini Pro model.
    *   [Synthetic Data Vault (SDV)](https://sdv.dev/): A Python library for generating synthetic data.
    *   [Pandas](https://pandas.pydata.org/): A powerful data manipulation and analysis library.
*   **Frontend:**
    *   [Streamlit](https://streamlit.io/): A fast and easy way to create data apps in Python.
*   **Authentication:**
    *   [Google Auth](https://google-auth.readthedocs.io/en/master/): Google Authentication Library for Python.

## Architecture

The application is composed of two main components:

1.  **Backend (FastAPI):**
    *   Exposes a RESTful API for designing the schema, synthesizing the data, and retrieving reports.
    *   Interacts with the Vertex AI API to communicate with the Gemini Pro model for schema generation.
    *   Uses the SDV library to perform the data synthesis.
    *   Manages the state of the generation process using an in-memory cache.
2.  **Frontend (Streamlit):**
    *   Provides a user-friendly, step-by-step interface for the data generation process.
    *   Communicates with the backend API to trigger the different stages of the process.
    *   Displays previews of the generated schema, seed data, and final synthetic data.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/V1--Matrix-AI.git
    cd V1--Matrix-AI
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Google Cloud authentication:**
    *   Ensure you have a Google Cloud project with the Vertex AI API enabled.
    *   Authenticate your environment using the Google Cloud CLI:
        ```bash
        gcloud auth application-default login
        ```

4.  **Run the application:**
    ```bash
    ./run.sh
    ```
    This will start both the backend and frontend servers.

## Usage

1.  **Open the web interface:**
    *   Navigate to `http://localhost:8501` in your web browser.

2.  **Step 1: Design Schema:**
    *   Provide a natural language description of the dataset you want to create.
    *   Specify the number of records to generate.
    *   Click "Design Schema" to have the AI generate the metadata and seed data.

3.  **Step 2: Review and Synthesize:**
    *   Review the AI-generated schema and seed data.
    *   If satisfied, provide your email address for notifications and click "Synthesize Data."

4.  **Step 3: Monitor and Review:**
    *   Monitor the progress of the data synthesis.
    *   Once complete, you will receive an email notification.
    *   Review samples of the generated data in the web interface.

5.  **Step 4: Download:**
    *   Download the complete synthetic dataset and reports as a ZIP file.

## Project Structure

```
/home/pavan_thalla/dev/V1--Matrix-AI/
├───.dockerignore
├───.gitignore
├───Dockerfile
├───requirements.txt
├───run.sh
├───backend/
│   ├───main.py             # FastAPI application entry point
│   └───core/
│       ├───ai.py           # Logic for interacting with the Gemini AI model
│       ├───analytics.py    # Data analysis and reporting functions
│       ├───cache.py        # In-memory cache for storing job status
│       ├───config.py       # Configuration settings
│       ├───google_auth.py  # Google Cloud authentication helper
│       ├───notifications.py# Email notification logic
│       ├───schemas.py      # Pydantic schemas for API requests and responses
│       └───sdv.py          # Synthetic Data Vault (SDV) generation logic
├───frontend/
│   ├───api.py              # Functions for making API calls to the backend
│   ├───app.py              # Main Streamlit application file
│   ├───styles.py           # CSS styles for the Streamlit app
│   └───components/
│       ├───sidebar.py      # Renders the sidebar in the Streamlit app
│       └───steps.py        # Renders the different steps in the UI
└───frontend_v1/            # (Older version of the frontend)
    ├───api.py
    ├───app.py
    ├───styles.py
    └───components/
        ├───sidebar.py
        ├───stepper.py
        └───steps.py
```
