import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import uuid
import logging
import json
import gc

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
    BASE_DIR = Path(__file__).parent.resolve()
    STATIC_DIR = BASE_DIR / "static"

from sharp.models import PredictorParams, create_predictor
from sharp.models.params import InitializerParams
from sharp.utils import io
from sharp.utils.gaussians import unproject_gaussians, save_ply as original_save_ply, Gaussians3D

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

# Configuration
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def init_model(stride=2):
    LOGGER.info(f"Initializing model on device: {DEVICE} (stride={stride})")
    # Force model cache to be local to the EXE directory to avoid cluttering user home dir
    model_cache = BASE_DIR / "model_cache"
    model_cache.mkdir(exist_ok=True)
    os.environ["TORCH_HOME"] = str(model_cache)
    try:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        params = PredictorParams()
        params.initializer.stride = stride
        params.gaussian_decoder.stride = stride
        predictor = create_predictor(params)
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(DEVICE)
        LOGGER.info("Model initialized successfully.")
        return predictor
    except Exception as e:
        LOGGER.error(f"Failed to initialize model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    import webbrowser
    import threading
    import time
    
    def open_browser():
        time.sleep(1.5)  # Wait a moment for the server to fully start up and bind to port
        print("\n[SERVER] Opening browser to http://127.0.0.1:8000...")
        webbrowser.open("http://127.0.0.1:8000")
        
    threading.Thread(target=open_browser, daemon=True).start()

@app.post("/generate")
async def generate_3dgs(file: UploadFile = File(...), name: str = Form(None)):
    print(f"\n[SERVER] Received generation request for file: {file.filename}, custom name: {name}")
    
    # Save the uploaded file
    file_id = str(uuid.uuid4())
    input_ext = Path(file.filename).suffix.lower()
    
    # Create a unique directory for this session in uploads
    session_upload_dir = UPLOAD_DIR / file_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = session_upload_dir / f"input{input_ext}"
    
    # Create a unique directory for this session in output
    session_output_dir = OUTPUT_DIR / file_id
    session_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SERVER] Saving input to: {input_path}")
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)    
        
    try:
        is_video = input_ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        
        if is_video:
            print(f"[SERVER] Processing video file with enhanced extraction...")
            import imageio
            # Use ffmpeg plugin explicitly to avoid banding and ensure clean RGB24
            reader = imageio.get_reader(input_path, format='FFMPEG', mode='I')
            meta = reader.get_meta_data()
            fps = meta.get('fps', 10)
            
            # Extract first frame for overall metadata
            first_frame = reader.get_data(0)
            height, width = first_frame.shape[:2]
        else:
            from PIL import Image, ExifTags
            img_pil = Image.open(input_path)
            width, height = img_pil.size
            
        aspect_ratio = width / height

        # Robust EXIF Extraction (only for images)
        exif_data = {}
        f_raw = None
        f_35mm = None
        if not is_video:
            try:
                raw_exif = img_pil._getexif()
                if raw_exif:
                    for tag, value in raw_exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = value
            except Exception as e:
                print(f"[SERVER] Metadata extraction warning: {e}")

            def to_float(val):
                if val is None: return None
                try:
                    if isinstance(val, (int, float)): return float(val)
                    if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                        return float(val.numerator) / float(val.denominator)
                    if isinstance(val, tuple) and len(val) == 2:
                        return float(val[0]) / float(val[1])
                    return float(val)
                except: return None

            f_raw = to_float(exif_data.get("FocalLength"))
            f_35mm = to_float(exif_data.get("FocalLengthIn35mmFilm"))

        if not f_35mm:
            if f_raw:
                f_35mm = f_raw * 7.0 if f_raw < 15 else f_raw
            else:
                f_35mm = 35.0 # Default
        
        print(f"[SERVER] Initial Metadata: Res={width}x{height}, Aspect={aspect_ratio:.2f}, Focal35={f_35mm}mm")

        # Load AI Model
        stride = 2
        print(f"[SERVER] Loading AI Model to {DEVICE} with stride {stride}...")
        predictor = init_model(stride=stride)
        
        internal_shape = (1536, 1536)
        processed_plys = []
        
        if is_video:
            # Robust frame count
            try:
                num_frames = reader.get_length()
                if num_frames == float('inf') or num_frames <= 0:
                    num_frames = sum(1 for _ in reader)
                    reader = imageio.get_reader(input_path, format='FFMPEG', mode='I')
            except:
                num_frames = 30
                
            max_frames = 30
            step = max(1, num_frames // max_frames)
            
            print(f"[SERVER] Video length: {num_frames} frames. Processing up to {max_frames} (step={step})...")
            
            frame_idx_in_sequence = 0
            for i, frame in enumerate(reader):
                if frame_idx_in_sequence >= max_frames: break
                if i % step != 0: continue
                
                print(f"[SERVER] Processing frame {i} (Index {frame_idx_in_sequence})...")
                # Ensure frame is RGB and clean
                if frame.shape[2] == 4: frame = frame[:,:,:3]
                
                # Save the extracted frame to uploads session folder for reference
                frame_img_path = session_upload_dir / f"frame_{frame_idx_in_sequence:03d}.jpg"
                import PIL.Image
                PIL.Image.fromarray(frame).save(frame_img_path)
                
                # Predict with full splat count (decimate=1) to avoid banding artifacts
                gaussians = process_single_frame(predictor, frame, internal_shape, decimate=1)
                
                # Save PLY to output session folder
                frame_id = f"frame_{frame_idx_in_sequence:03d}"
                output_ply = session_output_dir / f"{frame_id}.ply"
                
                f_px = (f_35mm * np.sqrt(width**2 + height**2)) / np.sqrt(36**2 + 24**2)
                save_ply_standard(gaussians, f_px, (height, width), output_ply)
                
                # URL relative to the session folder
                processed_plys.append(f"/output/{file_id}/{frame_id}.ply")
                frame_idx_in_sequence += 1
                
            reader.close()
        else:
            # Single image process
            image, _, f_px = io.load_rgb(input_path)
            gaussians = process_single_frame(predictor, image, internal_shape)
            
            output_ply = session_output_dir / "model.ply"
            save_ply_standard(gaussians, f_px, (height, width), output_ply)
            processed_plys.append(f"/output/{file_id}/model.ply")

        # Unload model
        del predictor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        metadata = {
            "id": file_id,
            "type": "video" if is_video else "image",
            "focal_length_35mm": float(f_35mm),
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "filename": name if name else file.filename,
            "input_url": f"/uploads/{file_id}/input{input_ext}",
            "ply_urls": processed_plys,
            "fps": fps if is_video else 0,
            "session_id": file_id
        }
        
        output_json = OUTPUT_DIR / f"{file_id}.json"
        with open(output_json, "w") as f:
            json.dump(metadata, f)
        
        print(f"[SERVER] SUCCESS: Generation complete for {file_id}\n")
        return {
            "id": file_id, 
            "ply_url": processed_plys[0], 
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"[SERVER] CRITICAL ERROR during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def process_single_frame(predictor, image_np, internal_shape, decimate=1):
    """Helper to run predictor on a single numpy image frame."""
    h_orig, w_orig = image_np.shape[:2]
    
    f_35mm = 35.0
    f_px = (f_35mm * np.sqrt(w_orig**2 + h_orig**2)) / np.sqrt(36**2 + 24**2)
    
    image_pt = torch.from_numpy(image_np.copy()).float().to(DEVICE).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / w_orig]).float().to(DEVICE)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    with torch.no_grad():
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
        
        intrinsics = torch.tensor([
            [f_px, 0, w_orig / 2, 0],
            [0, f_px, h_orig / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]).float().to(DEVICE)
        
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / w_orig
        intrinsics_resized[1] *= internal_shape[1] / h_orig

        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(DEVICE), intrinsics_resized, internal_shape
        )
        
        # Apply decimation if requested
        if decimate > 1:
            gaussians = Gaussians3D(
                mean_vectors=gaussians.mean_vectors[:, ::decimate, :],
                singular_values=gaussians.singular_values[:, ::decimate, :],
                quaternions=gaussians.quaternions[:, ::decimate, :],
                colors=gaussians.colors[:, ::decimate, :],
                opacities=gaussians.opacities[:, ::decimate]
            )
            print(f"[SERVER] Decimated Gaussians to {gaussians.mean_vectors.shape[1]} splats")
            
    return gaussians

@app.get("/output/{file_id}/{filename}")
async def get_output_file(file_id: str, filename: str):
    file_path = OUTPUT_DIR / file_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/history")
async def list_history():
    files = []
    if OUTPUT_DIR.exists():
        # Each generation has a {file_id}.json in the root of OUTPUT_DIR
        for json_file in OUTPUT_DIR.glob("*.json"):
            mtime = json_file.stat().st_mtime
            try:
                with open(json_file, "r") as f:
                    metadata = json.load(f)
                
                files.append({
                    "id": json_file.stem,
                    "ply_url": metadata.get("ply_urls", [f"/output/{json_file.stem}/model.ply"])[0],
                    "timestamp": mtime,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"[SERVER] Error loading metadata from {json_file}: {e}")
    
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files

@app.delete("/delete/{file_id}")
async def delete_item(file_id: str):
    print(f"[SERVER] Deleting session: {file_id}")
    try:
        # Paths
        session_upload_dir = UPLOAD_DIR / file_id
        session_output_dir = OUTPUT_DIR / file_id
        json_path = OUTPUT_DIR / f"{file_id}.json"
        
        # Recursive delete
        if session_upload_dir.exists():
            shutil.rmtree(session_upload_dir)
        if session_output_dir.exists():
            shutil.rmtree(session_output_dir)
        if json_path.exists():
            os.remove(json_path)
        
        return {"status": "success"}
    except Exception as e:
        print(f"[SERVER] Error deleting session {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
