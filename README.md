# Matrix AI Generator

A sophisticated synthetic data generation service that combines AI-powered schema generation with the Synthetic Data Vault (SDV) framework.

## Project Overview

This service provides a FastAPI-based web API that generates synthetic data based on natural language descriptions. It leverages Google Cloud Platform services, particularly Vertex AI's Gemini model for intelligent schema generation and Google Cloud Storage for data persistence.

## Key Features

- **AI-Powered Schema Generation**: Uses Vertex AI (Gemini) to convert natural language descriptions into database schemas
- **Synthetic Data Generation**: Employs SDV framework to create realistic synthetic data
- **Cloud Storage Integration**: Automatically stores generated datasets in Google Cloud Storage
- **RESTful API**: Easy-to-use FastAPI endpoints for data generation requests

## Technology Stack

- **Backend Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **Data Processing**: Pandas
- **Synthetic Data Generation**: SDV (Synthetic Data Vault)
- **Data Validation**: Pydantic
- **Cloud Services**: 
  - Google Cloud Storage
  - Google Cloud Vertex AI (Gemini)
- **Container Support**: Docker

## Prerequisites

- Python 3.x
- Google Cloud Platform Account
- Enabled APIs:
  - Cloud Storage
  - Vertex AI
- Configured GCP credentials

## Configuration

Update `config.py` with your GCP settings:
```python
GCP_PROJECT_ID = "your-gcp-project-id"
GCS_BUCKET_NAME = "your-gcs-bucket-name"
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pavanthalla-a11y/Matrix-Ai.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Send a POST request to `/generate-and-store` with:
```json
{
    "data_description": "Your natural language description of the desired data",
    "num_records": 1000
}
```

The service will:
1. Generate appropriate database schema using AI
2. Create synthetic data based on the schema
3. Store the results in Google Cloud Storage
4. Return the status of the operation

## Docker Support

Build and run using Docker:
```bash
docker build -t matrix-ai-generator .
docker run -p 8000:8000 matrix-ai-generator
```

## License

[MIT License](LICENSE)

## Contributors

- Pavan Thalla
The Synthetic Data Generator Tool

Main Purpose: It's a FastAPI-based web service that generates synthetic data based on natural language descriptions using AI and stores it in Google Cloud Storage.

Key Components:

A FastAPI web service (main.py)
Integration with Google Cloud Platform (GCP)
Uses Vertex AI's Gemini model for AI-powered schema generation
Uses SDV (Synthetic Data Vault) for synthetic data generation
Core Functionality:

Takes a natural language description of data and desired number of records
Uses Gemini AI to generate:
Database schema/metadata
Seed data for tables
Uses SDV to generate synthetic data based on the AI-generated schema
Uploads the generated data to Google Cloud Storage as CSV files
Technologies Used:

fastapi for the web API
uvicorn as the ASGI server
pandas for data manipulation
sdv for synthetic data generation
google-cloud-storage for GCS integration
google-cloud-aiplatform for Vertex AI (Gemini) integration
pydantic for data validation
Infrastructure:

Containerized application (indicated by presence of Dockerfile)
Configured to work with Google Cloud Platform
Configuration managed through config.py
