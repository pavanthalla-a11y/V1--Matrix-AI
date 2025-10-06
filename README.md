# Matrix AI - Synthetic Data Generator

Matrix AI is a high-performance, AI-powered synthetic data generation tool that leverages Google's Gemini Pro and the Synthetic Data Vault (SDV) library. It provides a streamlined, four-step process to describe, design, synthesize, and download realistic, tabular data for AI/ML model training, testing, and data augmentation.

For a detailed technical overview of the project, including architecture, data schemas, and API endpoints, please refer to the [Project Context](project_context.md) file.

## Features

*   **AI-Powered Schema Design:** Describe your desired dataset in natural language, and Matrix AI's Gemini-powered agent will automatically generate the relational schema and realistic seed data.
*   **High-Performance Synthesis:** Utilizes the Synthetic Data Vault (SDV) library to generate large-scale datasets based on the AI-generated schema. The process is optimized for performance with batch processing and faster synthesizers.
*   **Relational Data Support:** Generate complex, multi-table datasets with primary and foreign key relationships, ensuring referential integrity.
*   **Data Quality and Realism:** The AI agent is designed to produce high-quality, realistic data, including names, addresses, and other common data types.
*   **Interactive Web Interface:** A Streamlit-based web application guides users through the four-step data generation process.
*   **Asynchronous Processing:** Data synthesis runs as a background task, allowing users to monitor progress and receive email notifications upon completion.
*   **Comprehensive Reporting:** After synthesis, the tool provides detailed reports on data distribution and generation metrics.
*   **Easy Download:** Download the complete dataset, including all tables and reports, as a single ZIP file.

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