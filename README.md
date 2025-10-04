Agricultural Pest Detection & Recommendation System
This project is a complete, full-stack application designed to assist farmers by providing instant, AI-powered identification of agricultural pests and delivering actionable treatment advice. Users can upload an image of a pest, and the system will classify it using a state-of-the-art deep learning model, then provide detailed recommendations from a self-contained, authoritative knowledge base.

(Note: You can replace this with a real screenshot of your app)

Key Features
High-Accuracy Pest Classification: Utilizes a ConvNeXt-T model fine-tuned on the India-centric Pestopia dataset, achieving a Top-3 accuracy of over 90% on 132 pest classes.

Self-Contained Knowledge Base: All recommendations are served from an integrated SQLite database, making the application fast, reliable, and free to operate.

Interactive Web Application: A user-friendly frontend built with Streamlit that allows for easy image uploads and clear, structured results.

Robust Backend: A powerful Python backend handles the complex tasks of model inference and database queries.

Tech Stack
Component

Technology

Backend

Python, FastAPI, PyTorch, timm (for the model), SQLite (for the knowledge base)

Frontend

Python, Streamlit

ML Model

ConvNeXt-T pre-trained on ImageNet-21k, fine-tuned for fine-grained classification.

Training

Kaggle Notebooks with a T4 GPU, implementing RandAugment, Class-Balanced Focal Loss, and LLRD.

Project Structure
The project is organized in a monorepo structure with a clear separation of concerns:

pest-detection-system-final/
├── app/                  <-- The unified Streamlit application
│   └── app.py
├── backend/              <-- Contains the backend logic and database
│   ├── app/
│   │   ├── knowledge_base.db
│   │   └── ...
│   └── requirements.txt
├── models/               <-- Contains the trained AI model
│   └── convnext_pestopia_LLRD_best.pt
├── scripts/              <-- Utility scripts for data engineering
│   ├── database_setup.py
│   └── populate_db.py
└── README.md

Local Setup and Running the Application
Follow these steps to run the entire application on your local machine. You will need to have Python and Node.js installed.

1. Clone the Repository
First, clone the project to your local machine:

git clone [https://github.com/mohdirfan-code/pest-detection-system-final.git](https://github.com/mohdirfan-code/pest-detection-system-final.git)
cd pest-detection-system-final

2. Set Up the Backend Environment
The backend requires a Python virtual environment to manage its dependencies.

# Navigate to the backend folder
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Navigate back to the root directory
cd ..

3. Set Up the Frontend Environment
The Streamlit frontend also requires its own set of Python packages.

# Navigate to the frontend folder
cd frontend-streamlit

# Install the required Python packages
pip install streamlit pillow torch torchvision timm pandas

4. Run the Application
The application requires two separate terminals to run.

Terminal 1: Start the Backend API

# Navigate to the backend folder
cd backend

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Run the FastAPI server
uvicorn app.main:app --reload

You should see a message confirming the server is running on http://127.0.0.1:8000.

Terminal 2: Start the Frontend App

# Navigate to the frontend-streamlit folder
cd app

# Run the Streamlit application
streamlit run app.py

A new tab will open in your web browser with the Streamlit application. It is now fully connected to your backend.
