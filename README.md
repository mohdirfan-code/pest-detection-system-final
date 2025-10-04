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
cd frontend-streamlit

# Run the Streamlit application
streamlit run app.py

A new tab will open in your web browser with the Streamlit application. It is now fully connected to your backend.
