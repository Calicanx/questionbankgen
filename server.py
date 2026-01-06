import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bson import ObjectId

from questionbank.core.generator import QuestionGenerator
from questionbank.mongodb.repository import question_repo
from questionbank.llm.gemini_client import GENERATED_IMAGES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QuestionBank API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
app.mount("/static/generated_images", StaticFiles(directory=GENERATED_IMAGES_DIR), name="generated_images")

class GenerationRequest(BaseModel):
    source_id: Optional[str] = None
    widget_type: Optional[str] = "image"
    variation_type: str = "number_change"

@app.get("/api/generated")
async def get_generated_questions(limit: int = 50):
    """Fetch already generated questions and their sources."""
    try:
        # Get latest generated questions
        generated_docs = list(question_repo.generated.find().sort("metadata.generated_at", -1).limit(limit))
        
        results = []
        for gen_doc in generated_docs:
            source_id = gen_doc.get("source_question_id")
            if source_id:
                source = question_repo.get_question_by_id(str(source_id))
                if source:
                    source["_id"] = str(source["_id"])
                    
                    # Ensure generated data is in the expected format for frontend
                    gen_data = gen_doc.get("perseus_json", {})
                    gen_data["_id"] = str(gen_doc["_id"])
                    gen_data["source_id"] = str(source_id)
                    
                    results.append({
                        "id": str(gen_doc["_id"]),
                        "source": source,
                        "generated": gen_data,
                        "metadata": gen_doc.get("metadata", {})
                    })
        
        return results
    except Exception as e:
        logger.error(f"Error fetching generated questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_question(request: GenerationRequest):
    """Generate a question with images and return both source and generated."""
    try:
        generator = QuestionGenerator()
        
        if request.source_id:
            source = question_repo.get_question_by_id(request.source_id)
        else:
            source = question_repo.get_random_question(widget_type=request.widget_type)
            
        if not source:
            raise HTTPException(status_code=404, detail="Source question not found")
        
        source_id_str = str(source["_id"])
        source["_id"] = source_id_str
        
        # Check if we already have a generated version to avoid real-time generation if possible
        existing = question_repo.generated.find_one({"source_question_id": ObjectId(source_id_str)})
        if existing:
            generated = existing["perseus_json"]
            generated["_id"] = str(existing["_id"])
        else:
            # Generate if not exists
            generated = generator.generate_with_images(
                source_question=source,
                save_to_db=True,
                generate_new_images=True
            )
        
        if not generated:
            raise HTTPException(status_code=500, detail="Failed to generate question")
            
        return {
            "source": source,
            "generated": generated
        }
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/regenerate")
async def regenerate_question(request: GenerationRequest):
    """Force regenerate a question even if it already exists."""
    try:
        generator = QuestionGenerator()
        
        if not request.source_id:
            raise HTTPException(status_code=400, detail="Source ID is required for regeneration")
            
        source = question_repo.get_question_by_id(request.source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source question not found")
            
        source_id_str = str(source["_id"])
        source["_id"] = source_id_str
        
        # Force generation regardless of existing
        generated = generator.generate_from_source(
            source_question=source,
            variation_type=request.variation_type,
            save_to_db=True
        )
        
        if not generated:
            raise HTTPException(status_code=500, detail="Failed to regenerate question")
            
        return {
            "source": source,
            "generated": generated
        }
    except Exception as e:
        logger.error(f"Error regenerating question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

