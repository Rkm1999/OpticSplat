import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import uuid
import logging
import json

# Add ml-sharp/src to sys.path
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    BUNDLE_DIR = Path(sys._MEIPASS)
    sys.path.append(str(BUNDLE_DIR.absolute()))
    # In one-file mode, the EXE location is sys.executable
    BASE_DIR = Path(sys.executable).parent
    STATIC_DIR = BUNDLE_DIR / "static"
else:
    # Running in normal development
    sys.path.append(str(Path("ml-sharp/src").absolute()))
    BASE_DIR = Path(".")
    STATIC_DIR = Path("static")

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import unproject_gaussians, save_ply as original_save_ply

# Compatible PLY saving
def save_ply_standard(gaussians, f_px, image_shape, path):
    """Save a predicted Gaussian3D to a PLY file compatible with standard viewers."""
    from plyfile import PlyData, PlyElement
    
    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1).detach().cpu().numpy()
    num_gaussians = xyz.shape[0]
    
    # Standard 3DGS PLY expects nx, ny, nz (usually 0)
    normals = np.zeros((num_gaussians, 3), dtype=np.float32)
    
    # f_dc properties
    # SHARP predicts colors, we convert them to spherical harmonics (degree 0)
    # The original save_ply does some linearRGB to sRGB conversion
    from sharp.utils.gaussians import convert_rgb_to_spherical_harmonics
    from sharp.utils import color_space as cs_utils
    
    colors_srgb = cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    f_dc = convert_rgb_to_spherical_harmonics(colors_srgb).detach().cpu().numpy()
    
    # f_rest (degree 1-3) - 45 properties, set to 0
    f_rest = np.zeros((num_gaussians, 45), dtype=np.float32)
    
    # Opacity (logit)
    opacity = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1).detach().cpu().numpy()
    
    # Scale (log)
    scale = torch.log(gaussians.singular_values).flatten(0, 1).detach().cpu().numpy()
    
    # Rotation (quaternion)
    rotation = gaussians.quaternions.flatten(0, 1).detach().cpu().numpy()
    
    # Combine all into a structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    elements = np.empty(num_gaussians, dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        elements[f'f_rest_{i}'] = f_rest[:, i]
    elements['opacity'] = opacity[:, 0]
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)
    print(f"[SERVER] Exported standard 3DGS PLY with {num_gaussians} Gaussians")

# Set up logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI()

# Add COOP and COEP headers for SharedArrayBuffer support
@app.middleware("http")
async def add_coop_coep_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# Configuration
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Global model state
predictor = None

def init_model():
    global predictor
    LOGGER.info(f"Initializing model on device: {DEVICE}")
    # Force model cache to be local to the EXE directory to avoid cluttering user home dir
    model_cache = BASE_DIR / "model_cache"
    model_cache.mkdir(exist_ok=True)
    os.environ["TORCH_HOME"] = str(model_cache)
    try:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(DEVICE)
        LOGGER.info("Model initialized successfully.")
    except Exception as e:
        LOGGER.error(f"Failed to initialize model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    init_model()

@app.post("/generate")
async def generate_3dgs(file: UploadFile = File(...), name: str = Form(None)):
    print(f"\n[SERVER] Received generation request for file: {file.filename}, custom name: {name}")
    if not predictor:
        print("[SERVER] Error: Model not initialized!")
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Save the uploaded file
    file_id = str(uuid.uuid4())
    input_ext = Path(file.filename).suffix
    input_path = UPLOAD_DIR / f"{file_id}{input_ext}"
    
    print(f"[SERVER] Saving input to: {input_path}")
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load image for processing
        print(f"[SERVER] Loading image and extracting metadata...")
        from PIL import Image, ExifTags
        img_pil = Image.open(input_path)
        width, height = img_pil.size
        aspect_ratio = width / height

        # Robust EXIF Extraction
        exif_data = {}
        try:
            raw_exif = img_pil._getexif()
            if raw_exif:
                for tag, value in raw_exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
        except Exception as e:
            print(f"[SERVER] Metadata extraction warning: {e}")

        # Determine Focal Length (35mm equivalent)
        def to_float(val):
            if val is None: return None
            try:
                if isinstance(val, (int, float)): return float(val)
                # Handle PIL's IFDRational or tuple (numerator, denominator)
                if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                    return float(val.numerator) / float(val.denominator)
                if isinstance(val, tuple) and len(val) == 2:
                    return float(val[0]) / float(val[1])
                return float(val)
            except:
                return None

        f_raw = to_float(exif_data.get("FocalLength"))
        f_35mm = to_float(exif_data.get("FocalLengthIn35mmFilm"))

        if not f_35mm:
            if f_raw:
                # If raw focal is small (< 15), it's likely a phone/drone sensor (approx 6-8x crop)
                f_35mm = f_raw * 7.0 if f_raw < 15 else f_raw
            else:
                f_35mm = 35.0 # Default
        
        # Calculate Crop Factor
        crop_factor = 1.0
        if f_raw and f_35mm and f_raw > 0:
            crop_factor = f_35mm / f_raw

        print(f"[SERVER] Metadata Found: Res={width}x{height}, Aspect={aspect_ratio:.2f}, Focal35={f_35mm}mm, Crop={crop_factor:.2f}x")

        # Now pass to ml-sharp for AI processing
        image, _, f_px = io.load_rgb(input_path)
        # (We use ml-sharp's internal f_px for unprojection, but our f_35mm for UI)
        
        # Run inference
        internal_shape = (1536, 1536)
        print(f"[SERVER] Preprocessing image to {internal_shape}...")
        image_pt = torch.from_numpy(image.copy()).float().to(DEVICE).permute(2, 0, 1) / 255.0
        _, h, w = image_pt.shape
        disparity_factor = torch.tensor([f_px / w]).float().to(DEVICE)

        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        print(f"[SERVER] Running AI Inference on {DEVICE} (SHARP)...")
        with torch.no_grad():
            gaussians_ndc = predictor(image_resized_pt, disparity_factor)

            intrinsics = torch.tensor([
                [f_px, 0, w / 2, 0],
                [0, f_px, h / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).float().to(DEVICE)
            
            intrinsics_resized = intrinsics.clone()
            intrinsics_resized[0] *= internal_shape[0] / w
            intrinsics_resized[1] *= internal_shape[1] / h

            print(f"[SERVER] Unprojecting Gaussians to metric space...")
            gaussians = unproject_gaussians(
                gaussians_ndc, torch.eye(4).to(DEVICE), intrinsics_resized, internal_shape
            )

        # Save PLY and Metadata
        output_ply = OUTPUT_DIR / f"{file_id}.ply"
        output_json = OUTPUT_DIR / f"{file_id}.json"
        print(f"[SERVER] Saving compatible 3D Splat file (PLY) to: {output_ply}")
        save_ply_standard(gaussians, f_px, (height, width), output_ply)
        
        # Calculate f_35mm equivalent for the frontend simulator
        diag_35mm = np.sqrt(36**2 + 24**2)
        diag_px = np.sqrt(width**2 + height**2)
        f_35mm = f_px * diag_35mm / diag_px

        metadata = {
            "id": file_id,
            "focal_length_35mm": float(f_35mm),
            "focal_length_raw": float(f_raw) if f_raw else None,
            "crop_factor": float(crop_factor),
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "filename": name if name else file.filename,
            "input_url": f"/uploads/{input_path.name}"
        }
        
        with open(output_json, "w") as f:
            json.dump(metadata, f)
        
        print(f"[SERVER] SUCCESS: Generation complete for {file_id}\n")
        return {
            "id": file_id, 
            "ply_url": f"/output/{file_id}.ply",
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"[SERVER] CRITICAL ERROR during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/output/{filename}")
async def get_output(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/history")
async def list_history():
    files = []
    if OUTPUT_DIR.exists():
        for ply_file in OUTPUT_DIR.glob("*.ply"):
            mtime = ply_file.stat().st_mtime
            metadata = {}
            json_file = ply_file.with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        metadata = json.load(f)
                except:
                    pass
            
            files.append({
                "id": ply_file.stem,
                "ply_url": f"/output/{ply_file.name}",
                "timestamp": mtime,
                "metadata": metadata
            })
    
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files

@app.delete("/delete/{file_id}")
async def delete_item(file_id: str):
    print(f"[SERVER] Deleting item: {file_id}")
    try:
        # Paths
        ply_path = OUTPUT_DIR / f"{file_id}.ply"
        json_path = OUTPUT_DIR / f"{file_id}.json"
        
        # We need to find the upload file path from JSON metadata before deleting it
        if json_path.exists():
            with open(json_path, "r") as f:
                meta = json.load(f)
                input_url = meta.get("input_url")
                if input_url:
                    upload_filename = input_url.split("/")[-1]
                    upload_path = UPLOAD_DIR / upload_filename
                    if upload_path.exists():
                        os.remove(upload_path)
        
        # Delete PLY and JSON
        if ply_path.exists(): os.remove(ply_path)
        if json_path.exists(): os.remove(json_path)
        
        return {"status": "success"}
    except Exception as e:
        print(f"[SERVER] Error deleting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
