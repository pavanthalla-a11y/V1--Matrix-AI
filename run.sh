#!/bin/bash

# Start the backend server
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
cd ..

# Start the frontend server
cd frontend
streamlit run app.py --server.port 8501
