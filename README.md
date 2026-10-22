# HAWK-EYE: Crowd Monitoring, Counting & Density Estimation

A computer-vision project that uses **YOLOv8n** and **OpenCV** to detect people in video frames, estimate the visible person count, and present crowd-level trends in a dashboard.


## 1. Overview

HAWK-EYE processes a video or webcam stream frame by frame. For each processed frame, the detector predicts person bounding boxes. The application filters detections, counts the retained person boxes, and can record the count over time for visual monitoring.

### Core workflow

```text
Video file / Webcam
        |
        v
OpenCV reads a frame
        |
        v
YOLOv8n predicts objects
        |
        v
Keep detections labelled "person"
        |
        v
Count retained person boxes
        |
        v
(Optional extension) Map count to a crowd level
        |
        v
Display video, count and count trend
```

**Important distinction:** the number of person boxes in one frame is a *frame-level visible count*. It's uses  BYTETRACK  that automatically assign id number to the unique people who appeared throughout the video. 
## 2. Objectives

- Detect people in video frames using YOLOv8n.
- Draw bounding boxes around accepted person detections.
- Estimate the number of people visible in each processed frame.
- Track how the observed count changes over time.
- Present results in an accessible monitoring interface.
- Explore a count-based crowd-level feature.
- Provide a foundation for evaluating detection quality, counting error, and inference speed.

## 3. Technologies

| Technology | Role |
|---|---|
| Python | Main programming language |
| Ultralytics YOLOv8 | Object-detection model and inference interface |
| OpenCV (`cv2`) | Video capture, frame reading, image display and drawing utilities |
| Streamlit  | Web-based user interface |
| Pandas  | Tabular data handling and CSV export |
| Matplotlib or Plotly ) | Count-over-time visualisation |

The exact dependencies and versions should be taken from the project's environment or `requirements.txt`. The table above describes the intended roles and does not guarantee every library is currently used in the code.

## 4. How the system works

### Step 1  Video input

The user selects a video file or opens a webcam. OpenCV obtains frames from the selected source.

### Step 2  Person detection

YOLOv8n analyses each frame and predicts bounding boxes, class labels, and confidence scores. The application retains detections labelled as `person`, subject to the configured confidence threshold.

### Step 3  Frame-level counting

The visible count for a frame is the number of retained person detections:

```python
person_count = number_of_accepted_person_detections
```

This is a detection count, so occlusion, missed detections, false positives, and overlapping people can affect the estimate.

### Step 4  Visualisation

The application can display the annotated frame and the current count. If count history is recorded, it can also show a time-series plot and summary statistics such as average or peak observed count.

### Step 5  Crowd-level estimation

A proposed extension converts the frame count into a configurable label such as `LOW`, `MEDIUM`, or `HIGH`. Example thresholds must be tuned and evaluated for the specific camera and scene; they are not universal safety thresholds.

## 5. Crowd-density feature

### Count-based crowd level

A first implementation can use two configurable thresholds, `T1` and `T2`:

```text
If count < T1:
    level = LOW
Else if count < T2:
    level = MEDIUM
Else:
    level = HIGH
```

Choose the thresholds according to the camera view, expected occupancy, and validation results. Consider smoothing the count over several frames or requiring a level to persist before displaying an alert, so brief detection fluctuations do not make the label flicker.

### Count-based level is not physical density

A label derived only from the number of detections is **not** a physical density measurement. Physical density is usually expressed in people per square metre and requires a meaningful estimate of the real-world area represented in the image, plus appropriate camera geometry or calibration. Do not present uncalibrated count thresholds as crowd-safety limits.

## 6. YOLOv8n architecture : brief explanation

YOLOv8n is the nano-sized variant of YOLOv8, selected as a lightweight model option for inference.

- **Backbone:** extracts visual features from the input frame.
- **CBS / Conv block:** commonly describes convolution, batch normalisation, and SiLU activation. Convolution learns spatial patterns, batch normalisation supports training stability, and SiLU introduces non-linearity.
- **C2f:** uses split paths and intermediate feature processing/concatenation to encourage feature reuse and efficient gradient flow.
- **SPPF (Spatial Pyramid Pooling Fast):** applies sequential max-pooling operations and combines the resulting features to provide broader spatial context.
- **Neck:** fuses features at multiple scales so information from different spatial resolutions can support detection.
- **Detection head:** predicts class scores and bounding-box geometry at multiple scales.
- **Post-processing:** confidence filtering and non-maximum suppression (NMS), where configured by the inference pipeline, reduce weak or redundant detections.

These modules support object detection; the application-level counting step is performed after person detections have been retained.

## 7. Setup

### Prerequisites

- Python version supported by the installed Ultralytics release.
- `pip` and a virtual environment.
- A webcam or a video file for testing.
- A compatible CPU or NVIDIA GPU. A GPU is optional, but can improve inference speed.

### Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks script activation, you may use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**Linux / Ubuntu:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If no dependency file exists yet, install the core packages:

```bash
pip install ultralytics opencv-python
```

Install the dashboard and plotting libraries only if the application code imports them. For reproducible results, record the working versions in `requirements.txt`.

## 8. Running the project

The exact launch command depends on the entry-point filename in your repository. Common patterns are:

### Python script

```bash
python main.py
```

Replace `main.py` with the actual script name.

### Streamlit application

```bash
streamlit run app.py
```

Use this only if the project has a Streamlit entry point named `app.py`. Otherwise, replace it with the actual filename or follow the project's existing run instructions.

## 9. Suggested project structure

Your repository may use a different layout. A clean example is:

```text
HAWK-EYE/
├── app.py                  # Dashboard entry point (if using Streamlit)
├── main.py                 # Optional standalone inference script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── models/
│   └── yolov8n.pt          # Model weights (downloaded or supplied separately)
├── videos/
│   └── sample_video.mp4    # Optional test clip; avoid committing large files
└── outputs/
    ├── annotated_video.mp4 # Optional processed video
    └── counts.csv          # Optional frame-level count log
```

Do not create duplicate entry points just to match this example. Keep the structure consistent with the files in your actual project.

## 10. Evaluation

Evaluate the detector and the counting application separately.

| Goal | Suggested measure | What it tells you |
|---|---|---|
| Person detection | Precision, recall, mAP | How well people are detected |
| Counting | Mean Absolute Error (MAE) | Average difference between predicted and manually annotated counts |
| Processing speed | FPS or milliseconds per frame | How quickly frames are processed |
| Density-level stability | Level changes over time | Whether the label fluctuates because of noisy detections |

For frame-level counting, if `predicted_count_i` is the system count and `ground_truth_count_i` is a manually verified count for frame `i`, then:

```text
MAE = mean(abs(predicted_count_i - ground_truth_count_i))
```

Use representative test videos containing different crowd sizes, viewing distances, lighting conditions, occlusion levels, and camera angles. Report measured values only; do not invent performance results.

## 11. Limitations

- Occlusion can hide people from the detector.
- Small or distant people may be missed.
- False detections can inflate counts.
- A per-frame count may fluctuate even when the real crowd is stable.
- Without tracking, the system does not identify unique individuals across time.
- Count-based levels do not measure physical density or establish whether a location is safe.
- Performance depends on video quality, model weights, confidence settings, hardware, and input resolution.

## 12. Future improvements

- Add configurable crowd-level thresholds and temporal smoothing.
- Add alerts for sustained high-count periods.
- Calibrate a scene to estimate people per square metre where appropriate.
- Evaluate on labelled videos and publish counting error and inference speed.
- Add CSV export and session summaries if these are not already implemented.
- Test the system on different camera positions and lighting conditions.

## 13. Responsible use

Use video streams only where you have appropriate permission and follow applicable privacy rules. Prefer processing that does not identify individuals, restrict access to stored videos, and avoid presenting prototype outputs as authoritative safety decisions.

## 14. Acknowledgements and references

- **Ultralytics YOLO documentation:** https://docs.ultralytics.com/
- **OpenCV documentation:** https://docs.opencv.org/

Refer to the installed model version and official documentation for exact implementation details.

---

