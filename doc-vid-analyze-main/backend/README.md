# 📄 Doc-Vid-Analyze Backend

This is the backend API for the Doc-Vid-Analyze project. It provides endpoints for analyzing legal documents, videos, and audio, extracting insights, and managing user authentication and subscriptions.

---

## 🚀 Features

- Document analysis (PDF)
- Video and audio transcription and analysis
- Legal question answering (chatbot)
- Risk assessment and visualization
- Contract clause analysis
- User authentication and subscription management (with PayPal integration)
- Tiered access (Free, Standard, Premium)

---

## 🛠️ Tech Stack

- **FastAPI** - API framework
- **spaCy / Transformers / SentenceTransformers** - NLP processing
- **MoviePy** - Video processing
- **pdfplumber** - PDF text extraction
- **SQLite** - Database
- **PayPal** - Subscription payments

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/HardikKandpal/doc_vid_analyzer.git
cd doc-vid-analyze-main/backend
```

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
 ```

### 3️⃣ (Optional) Download or Cache NLP Models

The backend will automatically download required models on first run. To speed up startup, you can pre-download models or place them in the models_cache directory.

## ▶️ Running the Backend

### Using Python

```bash
python app.py
 ```

## 📑 API Documentation

Once running, visit http://localhost:8000/docs for interactive API docs.

## 📁 Project Structure

- **app.py**: Entry point for the FastAPI app.
- **auth.py**: Handles authentication logic and user-related endpoints.
- **paypal_integration.py**: Contains logic for handling PayPal subscriptions and webhooks.
- **requirements.txt**: Lists all required Python packages for the backend.
- **models_cache/**: Directory to store preloaded or cached NLP models (optional use).
- **data/**: Contains the SQLite database or related files.

           

 # 📄 Doc-Vid-Analyze Frontend


## 📝 License
MIT License © 2025. Free to use and modify for research and non-commercial projects.



---

## Frontend README

# Legal Document & Video Analyzer Frontend

This is the frontend for the AI-powered Legal Document & Video Analyzer project. It provides a user-friendly interface for analyzing legal documents and videos, extracting insights, and getting AI-powered answers to legal questions.

---

## Features

- **Document Analysis**: Upload and analyze legal documents (PDFs) to extract key information, identify risks, and generate summaries.
- **Video Analysis**: Upload videos to transcribe speech and analyze the content for legal insights.
- **Legal Q&A**: Ask questions about your analyzed documents and get AI-powered answers.
- **Risk Visualization**: View interactive charts and visualizations of legal risks in your documents.

---

## Technologies Used

- **React**: Frontend framework
- **Material UI**: Component library for modern UI design
- **Axios**: HTTP client for API requests
- **React Router**: For navigation between pages
- **Chart.js**: For data visualization

---

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HardikKandpal/doc_vid_analyzer.git
    cd doc-vid-analyze-main/frontend
   ```

2. **Install dependencies** :

   ```bash
   cd frontend
   npm install
    ```
3.  Configure the backend URL :
   
   - Open src/config.js
   - Update the API_BASE_URL with your backend URL (e.g., http://localhost:8000 )
4. Start the development server :
   
   ```bash
   npm start
   ```
5. Build for production :
   
   ```bash
   npm run build
    ```
    or 
    ```bash
    npm run dev
    ```
## Backend Connection
This frontend connects to a FastAPI backend that provides the following services:

- Document analysis (PDF)
- Video analysis and transcription
- Legal question answering
- Risk visualization
Make sure the backend is running and accessible at the URL specified in src/config.js .

## Project Structure

- **public/**: Contains static assets like `index.html`, images, and other publicly accessible files.
- **src/**: The main source folder for the React frontend.
  - **components/**: Houses reusable UI components like buttons, headers, etc.
  - **pages/**: Defines route-level components (e.g., Home, Login).
  - **services/**: Contains API call logic and integration with backend endpoints.
  - **config.js**: Centralized configuration settings (e.g., base API URL).
  - **App.js**: Root component that sets up routing and global context.


## Usage
1. Document Analysis :
   
   - Navigate to the Document Analyzer page
   - Upload a PDF file
   - View the summary, risk assessment, and visualizations
2. Video Analysis :
   
   - Navigate to the Video Analyzer page
   - Upload a video file
   - View the transcript, summary, and risk assessment
3. Legal Q&A :
   
   - Navigate to the Legal Q&A page
   - Enter the Task ID from a previous analysis
   - Ask legal questions about the document or video
## License
MIT License © 2025. Free to use and modify for research and non-commercial projects.
