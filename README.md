# ArUco Marker Utility

This is a simple command-line utility to generate and detect ArUco markers using OpenCV. 

## Requirements

Ensure you have installed the required dependencies. You can install them by running:

```bash
pip install opencv-contrib-python numpy
```

## Usage

### 1. Generate an ArUco Marker

You can generate an ArUco marker by specifying its ID and optional size or output path.
The generated marker includes a white border padding to ensure it can be easily detected.

```bash
python aruco_tool.py generate --id 42 --out marker.png
```

Options:
- `--id`: The ID of the marker to generate (default: 1)
- `--size`: The size in pixels of the marker (default: 200)
- `--out`: Output image path (default: marker.png)

### 2. Detect ArUco Markers in an Image

You can detect ArUco markers in a static image file. It will draw bounding boxes and IDs over detected markers and save it as a new image prefixed with `detected_`.

```bash
python aruco_tool.py detect-img marker.png
```

Options:
- `image`: The path to the image you want to scan for ArUco markers.
- `--headless`: Run without attempting to open an image preview window. Useful for environments without a display.

### 3. Detect ArUco Markers using a Webcam

To open a live webcam feed and detect ArUco markers in real-time, run:

```bash
python aruco_tool.py detect-webcam
```

Options:
- `--camera`: The index of the camera to use (default: 0).

Press `q` to exit the webcam stream.
