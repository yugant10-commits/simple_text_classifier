from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from src.model_handler import get_prediction

router = APIRouter()

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

@router.post("/predict", response_model=Dict[str, str])
async def predict(input_data: TextInput):
    """
    Endpoint to classify a single news article.
    Returns the category (World, SPORTS, BUSINESS, SCI/TECH).
    """
    try:
        result = get_prediction(input_data.text)
        return {"category": result["category"], "confidence": str(result["confidence"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict", response_model=List[Dict[str, str]])
async def batch_predict(input_data: BatchTextInput):
    """
    Endpoint to classify multiple news articles at once.
    Returns the category for each article.
    """
    try:
        results = []
        for text in input_data.texts:
            result = get_prediction(text)
            results.append({"category": result["category"], "confidence": str(result["confidence"])})
        return results
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Endpoint to check if the API is running.
    """
    return {"status": "healthy"}