# main.py
# FastAPI Inference API for News Headline Classification

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI app
app = FastAPI(title="News Headline Classification API",
              description="API for classifying news headlines into World, Sports, Business, or Sci/Tech categories",
              version="1.0.0")

# Pydantic model for request validation
class HeadlineRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Global leaders meet to discuss climate change policies"
            }
        }

# Pydantic model for response
class PredictionResponse(BaseModel):
    category: str
    category_id: int
    confidence: float
    
    class Config:
        schema_extra = {
            "example": {
                "category": "World",
                "category_id": 0,
                "confidence": 0.92
            }
        }

# Load model, tokenizer and label mapping
MODEL_PATH = "./headline_classification_model"

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Global variables for model, tokenizer and label mapping
model = None
tokenizer = None
label_map = None

# Initialize model and tokenizer on startup
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, label_map
    
    try:
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load label mapping
        with open(f"{MODEL_PATH}/label_map.json", "r") as f:
            label_map = json.load(f)
        
        print("Model, tokenizer, and label mapping loaded successfully")
        print(f"Label map: {label_map}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a production environment, you'd want to raise an exception or handle this more gracefully
        # For now, we'll set up a placeholder label map for demonstration
        label_map = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
        raise RuntimeError(f"Failed to load model: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "headline_classification"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HeadlineRequest):
    # Check if model and tokenizer are loaded
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess the text
        processed_text = preprocess_text(request.text)
        
        # Tokenize
        inputs = tokenizer(
            processed_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Get category label
        category = label_map.get(str(prediction), label_map.get(prediction, "Unknown"))
        
        # Return response
        return {
            "category": category,
            "category_id": prediction,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "News Headline Classification API",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health"
        },
        "usage": "Send a POST request to /predict with a JSON body containing the 'text' field"
    }

# Run the API (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)