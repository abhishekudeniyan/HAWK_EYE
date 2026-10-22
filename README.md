# CrowdWatch

Crowd counting with YOLO and ByteTrack. Analyze an uploaded video or use a webcam.

## Run on Windows

```powershell
cd "D:\MY Projects\CrowdWatch_Mini_Project"
streamlit run app.py
```

The required Python packages are listed in `requirements.txt`. If they are already installed, there is no need to reinstall them. On first use, YOLO may download the selected model weights.

## Counting

ByteTrack keeps an ID for a person across nearby frames. The unique count is an estimate: missed detections or long occlusions can give someone a new ID. Set frame sampling to `1` for the best tracking.
