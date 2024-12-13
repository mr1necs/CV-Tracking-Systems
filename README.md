# CV-Tracking-Systems

## Overview

**CV-Tracking-Systems** is a repository showcasing examples of computer vision systems for real-time object detection and tracking. Using powerful tools such as OpenCV and YOLO, the projects demonstrate automated tracking in various scenarios. The repository is ideal for beginners and enthusiasts looking to explore object detection and tracking systems.

---

## Features

1. **YOLO-Based Object Tracking:**

   - Utilizes YOLO for object detection and OpenCV for tracking and visualization.
   - Tracks specific objects in video streams.

2. **HSV-Based Object Tracking:**

   - Tracks a HSV-Based Object Tracking in real-time using HSV color space and contour detection.
   - Highlights object trajectories using a deque buffer.

---

## Requirements

Ensure you have the following installed on your system:

- Python 3.8 or later
- OpenCV
- numpy
- imutils
- ultralytics
- torch

Alternatively, you can install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/mr1necs/CV-Tracking-Systems.git
   cd CV-Tracking-Systems
   ```
2. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### YOLO-Based Object Tracking


3. Run the YOLO-based object tracker:

   ```bash
   python yolo_object_tracker.py --device [device] --camera [video_path] --buffer [buffer_size]
   ```

   - `device`: Choose between `cpu`, `cuda`, or `mps` (default: `cpu`).
   - `camera`: Path to the video file or use webcam by default.
   - `buffer`: Maximum trajectory buffer size (default: 64).

4. Press `q` to exit the program.

### HSV-Based Object Tracking

3. Run the HSV-Based Object Tracker:

   ```bash
   python hsv_object_tracker.py --video [video_path] --buffer [buffer_size]
   ```

   - `video`: Path to the video file or use webcam by default.
   - `buffer`: Maximum trajectory buffer size (default: 32).

4. Press `q` to exit the program.

---

## Project Structure

```
CV-Tracking-Systems/
|-- yolo_object_tracker.py   # YOLO-based object tracker
|-- hsv_object_tracker.py    # HSV-based object tracker using HSV and contours
|-- requirements.txt         # List of dependencies
|-- README.md                # Project documentation
```

---

## Contribution

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request. Please ensure your code follows best practices and includes appropriate documentation.

---

## Acknowledgments

- **OpenCV**: For providing powerful tools for image processing and computer vision.
- **YOLO**: For state-of-the-art object detection.
- Contributors and the open-source community for inspiration and resources.

---

Happy tracking!

