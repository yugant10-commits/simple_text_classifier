import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

# labels
id2label = {0: "World", 1: "SPORTS", 2: "BUSINESS", 3: "SCI/TECH"}

model = None
tokenizer = None

def load_model():
    """
    Load the trained model and tokenizer.
    This is called only once when the first prediction is made.
    """
    global model, tokenizer
    
    try:
        # Load the model from your trained directory

        model_path = "models/checkpoint-15000"  
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def get_prediction(text):
    """
    Make a prediction for a single text input.
    
    Args:
        text (str): The news article text
        
    Returns:
        dict: Dictionary with predicted category and confidence
    """
    global model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        success = load_model()
        if not success:
            raise Exception("Failed to load model")
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        
    # Get the predicted class and confidence
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    predicted_class_idx = torch.argmax(predictions, dim=1).item()
    confidence = probabilities[0][predicted_class_idx].item()
    
    # Map index to label
    predicted_category = id2label[predicted_class_idx]
    
    return {
        "category": predicted_category,
        "confidence": round(confidence, 4)
    }