# QuestionBank AI Generator

This project automates the generation of educational questions using Google's Gemini AI. It scrapes existing questions, stores them in MongoDB, and generates new variations with preserved or AI-generated images.

## Features

- **AI Question Generation**: Creates new variations of math questions using Gemini 2.0.
- **Image Handling**: Automatically downloads and serves images locally (fixing "Access Denied" errors).
- **Widget Support**: Supports various Khan Academy widgets (numeric-input, radio, dropdown, etc.).
- **MongoDB Integration**: Stores source and generated questions in MongoDB (Atlas or local).

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB Database (Atlas or local)
- Google Gemini API Key

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Environment Setup**:
    - Copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Update `.env` with your `MONGODB_URI`, `MONGODB_DB_NAME`, and `GEMINI_API_KEY`.

3.  **Install Backend Dependencies**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Install Frontend Dependencies**:
    ```bash
    cd frontend
    npm install
    cd ..
    ```

## Running the Application

1.  **Start the Backend Server**:
    This server handles API requests and serves generated images.
    ```bash
    python server.py
    ```
    - API: `http://localhost:8001/api`
    - Images: `http://localhost:8001/static/generated_images/`

2.  **Start the Frontend**:
    In a new terminal:
    ```bash
    cd frontend
    npm run dev
    ```
    - Open `http://localhost:5174` in your browser.

## Generating Questions

You can generate questions using the CLI or verify all widget types.

**Generate random questions:**
```bash
source .venv/bin/activate
python -m questionbank.cli generate --random --count 5
```

**Verify/Generate for ALL widget types:**
```bash
source .venv/bin/activate
python generate_all_types.py
```

## Troubleshooting Images

If images are not loading:
1. Ensure the **backend server** is running (`python server.py`).
2. Check `questionbank/generated_images/` to see if files are being downloaded.
3. Check the browser console for specific errors.
