from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import shutil
import os
import tempfile
from backend.contour_logic import ContourExtractor
import json

app = FastAPI()

# Mount Static Files
# Serve static directory at /static
app.mount("/static", StaticFiles(directory="./backend/static"), name="static")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse('./backend/static/index.html')

extractor = ContourExtractor()

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # Create a temporary file to save the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process the image
        output_json_path = temp_path + ".json"
        success = extractor.process_image(temp_path, output_json_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process image")
            
        # Read the result
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                data = json.load(f)
            
            # Cleanup
            os.remove(temp_path)
            os.remove(output_json_path)
            
            return JSONResponse(content=data)
        else:
             raise HTTPException(status_code=500, detail="Output file not generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
