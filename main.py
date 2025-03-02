from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import HTMLResponse

# Import the API router
from routes.api import router as api_router

# Create FastAPI app
app = FastAPI(title="News Classification API", 
             description="API for classifying news articles into categories",
             version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    # Read the HTML file
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Include the API router
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)