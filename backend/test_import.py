import sys
import os
from pathlib import Path

# Add ml-sharp/src to sys.path
sys.path.append(str(Path("ml-sharp/src").absolute()))

try:
    import sharp
    from sharp.models import create_predictor, PredictorParams
    print("Successfully imported sharp!")
except ImportError as e:
    print(f"Failed to import sharp: {e}")
except Exception as e:
    print(f"Error: {e}")
