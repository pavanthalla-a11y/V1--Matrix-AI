# Matrix-Ai
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
