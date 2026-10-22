from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import shutil
import os
import tempfile
from backend.contour_logic import ContourExtractor
from backend import mechanics
import json
import io
import logging
import math
import warnings
from pathlib import Path
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SOURCE_PIXELS = 12_000_000
MAX_RESIZED_PIXELS = 2_000_000
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
RESIZE_WIDTH = 1000  # Keep in sync with contour_logic and mechanics.
from pydantic import BaseModel

app = FastAPI()

# Mount Static Files
# Serve static directory at /static
app.mount("/static", StaticFiles(directory="./backend/static"), name="static")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse('./backend/static/index.html')


async def read_valid_image(file: UploadFile) -> bytes:
    """Read a bounded amount of data and inspect actual image contents."""
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="Please upload an image.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB upload limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as img:
                if img.format not in SUPPORTED_IMAGE_FORMATS:
                    raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, WebP, or BMP image.")
                width, height = img.size
                if (
                    width < 1
                    or height < 1
                    or width * height > MAX_SOURCE_PIXELS
                    or RESIZE_WIDTH * max(1, int(RESIZE_WIDTH * height / width))
                    > MAX_RESIZED_PIXELS
                ):
                    raise HTTPException(status_code=413, detail="Image dimensions are too large for processing.")
                img.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(status_code=415, detail="Invalid or damaged image.") from None
    return contents


def validate_number(value: str, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"Invalid {name}.") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise HTTPException(status_code=422, detail=f"Invalid {name}.")
    return number


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await read_valid_image(file)
    try:
        # Unique extractor and temporary directory for each request.
        # The directory cleans up on both success and failure.
        with tempfile.TemporaryDirectory(prefix="solidify-") as directory:
            image_path = Path(directory) / "image"
            result_path = Path(directory) / "points.json"
            image_path.write_bytes(contents)
            extractor = ContourExtractor()
            if not extractor.process_image(str(image_path), str(result_path)) or not result_path.is_file():
                raise HTTPException(status_code=422, detail="No usable contour found in image.")
            return JSONResponse(content=json.loads(result_path.read_text(encoding="utf-8")))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Image contour processing failed")
        raise HTTPException(status_code=500, detail="Image processing failed.") from None


@app.post("/optimize")
async def optimize(
    file: UploadFile = File(...),
    handle_x: str = Form("0.5"),
    handle_y: str = Form("0.5"),
    target_mass: str = Form("0.8"),
    ball_mass: str = Form("0.2"),
):
    hx = validate_number(handle_x, "handle_x", 0.0, 1.0)
    hy = validate_number(handle_y, "handle_y", 0.0, 1.0)
    tm = validate_number(target_mass, "target_mass", 0.001, 100.0)
    bm = validate_number(ball_mass, "ball_mass", 0.0, 100.0)
    contents = await read_valid_image(file)
    try:
        result = mechanics.run_optimization(contents, hx, hy, target_mass_rifle=tm, m_ball=bm)
        if "error" in result:
            raise HTTPException(status_code=422, detail="Image could not be optimized.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Image optimization failed")
        raise HTTPException(status_code=500, detail="Image processing failed.") from None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
