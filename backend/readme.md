# OpticSplat (SHARP Camera Simulator)

## What it does

* **Single-image to 3D:** You just feed it one picture, and the PyTorch backend uses the SHARP AI model to predict depth and volume, spitting out a standard `.ply` 3D Gaussian Splat file.
* **Smart metadata extraction:** When you upload an image, the app automatically reads the EXIF data to figure out the original camera's focal length and crop factor so the 3D unprojection is as accurate as possible.
* **Browser-based 3D viewer:** You don't need any special desktop software to view the splats; they render directly in the browser using WebGL.
* **Virtual camera controls:** You have total control over the virtual lens. You can adjust the focal length, pan, tilt, roll, shift the camera around, and even tweak the alpha removal threshold to adjust the sharpness of the splats.
* **Photography guides:** I added toggles for composition overlays like the rule of thirds, a center crosshair, and different aspect ratio masks (like 16:9, 4:3, or 9:16) to help you compose your virtual photos.
* **Session history:** The app keeps track of the splats you've generated so you can easily switch back and forth between them using the sidebar gallery.

## Getting started

Before you start, make sure you have Python 3.8 or newer installed. A dedicated GPU is highly recommended for the AI processing, but it will fall back to your CPU if needed. 

Also, make sure the Apple `ml-sharp` repository is included or cloned into the root directory as `ml-sharp/src`, as the backend needs it to run the inference.

**Running the app**

I've included scripts to make starting the server as easy as possible. It will handle installing the Python dependencies and starting the backend.

If you are on Windows, just double-click the batch file:
```bat
run_simulator.bat
```
If you are on macOS or Linux, open your terminal and run the shell script:

```Bash
chmod +x run_simulator.sh
./run_simulator.sh
```

Once the server spins up, it should automatically open your default web browser to http://127.0.0.1:8000.

**How to use it**

1. Drag and drop any image (JPEG, PNG, or HEIF) into the browser window, or use the upload button.
2. Give it a moment to process. The server has to resize the image, run it through the AI model, and build the 3D file.
3. Once it loads, use the sidebar on the right to play with the focal length, move the camera around, or turn on the composition grids.

When you have an angle you like, hit the "Capture View" button to download a high-res screenshot of your viewport.

**A quick note on the first run**

The SHARP AI model weights are pretty heavy (a few gigabytes). The very first time you generate a 3D scene, the backend will download this model from Apple's servers and cache it locally in a `model_cache` folder. Because of this, your first generation will take a bit longer, but subsequent runs will be much faster.

If you ever need to free up hard drive space, you can safely delete the contents of the `uploads`, `output`, and `model_cache` folders.