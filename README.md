# QuestionBank - AI Question Generator

A system for generating educational question variations using AI, with support for Khan Academy Perseus format questions.

## Features

- **Question Generation**: Creates new question variations from source questions using Gemini AI
- **Image Generation**: Generates new images for questions (diagrams, graphs, place value blocks)
- **Google Cloud Storage**: Automatic upload of generated images to GCS for persistence
- **Smart Regeneration**: Re-generate questions with new numbers while strictly preserving educational context
- **Step-by-Step Solutions**: Generates solutions using SymPy symbolic math (Photomath approach)
- **Hint Generation**: AI-generated hints for questions without source hints
- **LaTeX Rendering**: Full KaTeX support including chemistry formulas (mhchem)

## Project Structure

```
questionbank/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   └── App.tsx             # Main React app with question comparison UI
│   └── package.json
├── questionbank/                # Python backend
│   ├── core/
│   │   ├── generator.py        # Main question generator (LLM + validation)
│   │   └── prompt_builder.py   # Prompt templates for AI generation
│   ├── intelligence/
│   │   ├── image_generator.py  # AI image generation using Gemini
│   │   ├── constraint_extractor.py
│   │   ├── coherence_validator.py
│   │   └── smart_generator.py
│   ├── utils/
│   │   ├── solution_generator.py  # SymPy-based step-by-step solver
│   │   ├── math_visualizer.py     # Matplotlib graph generation
│   │   ├── image_generator.py     # Programmatic image generation
│   │   └── khan_colors.py         # Khan Academy color schemes
│   ├── validation/
│   │   ├── pipeline.py         # Validation orchestration
│   │   ├── schema_validator.py # Perseus JSON schema validation
│   │   ├── latex_validator.py  # LaTeX syntax validation
│   │   └── answer_verifier.py  # Mathematical answer verification
│   ├── mongodb/
│   │   ├── connection.py       # MongoDB Atlas connection
│   │   └── repository.py       # Database operations
│   └── llm/
│       └── gemini_client.py    # Google Gemini API client
├── server.py                   # FastAPI server
├── .env.example               # Environment variables template
└── requirements.txt           # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB Atlas account (or local MongoDB)
- Google Gemini API key
- Google Cloud Service Account (for GCS access) with 'Storage Object Creator' role

### Backend Setup

```bash
# Clone and navigate to project
cd questionbank

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Environment Variables

Create a `.env` file with:

```env
# MongoDB
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/ai_tutor
MONGODB_DB_NAME=ai_tutor

# Google Gemini
GOOGLE_API_KEY=your_gemini_api_key
# or
GEMINI_API_KEY=your_gemini_api_key

# Google Cloud Storage
GCS_BUCKET_NAME=your_gcs_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

## Running the Application

### Start Backend Server

```bash
# From project root
python server.py
# Server runs at http://localhost:8001
```

### Start Frontend

```bash
cd frontend
npm run dev
# Frontend runs at http://localhost:5173 (or next available port)
```

### Access the Application

Open http://localhost:5173 in your browser to see:
- Side-by-side comparison of original and AI-generated questions
- Expandable hints section for both questions
- Comparison of original vs generated questions
- **Regenerate Button**: Click "Regenerate" to create a fresh variation with new numbers (context preserved)
- Step-by-Step solutions for generated math questions
- Images (hosted on GCS)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/generated` | GET | Get all generated questions |
| `/api/generate` | POST | Generate a new question variation |

### Generate Question

```bash
curl -X POST http://localhost:8001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"source_id": "source_question_id"}'
```

## Key Components

### 1. Question Generator (`questionbank/core/generator.py`)

Main orchestrator that:
- Calls Gemini to generate question variations
- Validates output against Perseus schema
- Generates new images when content changes
- Adds solution steps using SymPy
- Generates hints if source has none

### 2. Solution Generator (`questionbank/utils/solution_generator.py`)

Uses SymPy for symbolic math to generate step-by-step solutions:
- Extracts equations from question content
- Solves algebraically with intermediate steps
- Supports linear equations, proportions, fractions

### 3. Image Generator (`questionbank/intelligence/image_generator.py`)

Generates images using Gemini AI:
- Context-aware image generation
- Supports geometric diagrams, graphs, illustrations
- Saves to `questionbank/generated_images/`

### 4. Frontend (`frontend/src/App.tsx`)

React application with:
- Side-by-side question comparison
- KaTeX math rendering with mhchem for chemistry
- `processHintContent()` - Fixes Khan Academy's nested align blocks
- Widget rendering (images, radio buttons, numeric inputs)

## Recent Improvements

### LaTeX Rendering Fixes

The `processHintContent()` function handles Khan Academy's malformed LaTeX:
- Converts nested `\begin{align}` to `\begin{aligned}`
- Adds proper `$$` display math delimiters with newlines
- Enables remarkMath to detect display math blocks

### Hint Generation

Questions without source hints now get AI-generated hints:
- 2-3 progressive hints per question
- Context-aware based on question content and correct answer

### Chemistry Support

Added mhchem extension for chemistry formulas:
- `\ce{NaCl}` - Chemical formulas
- `\pu{g/mol}` - Units
- `\cancel{}` - Cancellation marks

### Google Cloud Storage Integration

- **Automatic Uploads**: Generated images (AI & programmatic) are automatically uploaded to GCS.
- **Fallback Mechanism**: If GCS upload fails (e.g., permissions), images fall back to local storage seamlessly.
- **Public URLs**: Returns public GCS URLs for easy embedding.

### Regenerate API & Features

- **Strict Context Preservation**: The regeneration prompt relies on strict rules to keep the educational scenario identical while changing numbers.
- **Smart Rephrasing**: The system rephrases narrative text slightly to avoid repetition while maintaining the exact logic.
- **Frontend Integration**: "Regenerate" button added to the UI for instant variation generation.

## Testing

### Verify Backend

```bash
# Health check
curl http://localhost:8001/api/health

# Get generated questions
curl http://localhost:8001/api/generated | python -m json.tool
```

### Verify Frontend

1. Open http://localhost:5173
2. Scroll through questions to see:
   - Images rendering correctly
   - Hints expanding with proper LaTeX
   - Solutions showing for math questions

## Troubleshooting

### LaTeX Not Rendering

- Check browser console for KaTeX errors
- Ensure `processHintContent()` is being called
- Verify `$$` delimiters have blank lines around them

### Images Not Loading

- Check server logs for 404 errors
- Verify `generated_images` directory exists
- Check image URLs in API response

### Hints Missing

- For source questions: Check if source has `hints` array
- For generated: Ensure `_generate_hints_if_missing()` is called
- Check server logs for "Generated X hints" messages

## Architecture Notes

### Why SymPy for Solutions?

Instead of asking AI to generate solutions (which can hallucinate), we use SymPy for accurate symbolic math solving. This follows the Photomath approach - parse the equation, solve it algebraically, and show each step.

### Why Programmatic Image Generation?

AI-generated graphs/diagrams are often mathematically inaccurate. We use:
- Matplotlib for function plots and coordinate graphs
- Custom generators for place value blocks, fractions
- Gemini only for contextual images (photos, illustrations)

### Khan Academy Widget Types

Supported widgets:
- `radio` - Multiple choice
- `numeric-input` - Number answers
- `image` - Images with `backgroundImage.url`
- `definition` - Tooltips
- `expression` - Math expressions

## Contributing

1. Keep code in appropriate module directories
2. Add logging with the `logger` module
3. Handle errors gracefully (don't break generation if hints fail)
4. Test changes with multiple question types (math, chemistry, reading)
