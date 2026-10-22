import time
from collections import deque
import os
import shutil
import tempfile
import cv2
import pandas as pd
import streamlit as st
import plotly.express as px
from person_tracking import load_person_model, track_people

st.set_page_config(page_title="HAWK-EYE | Human Counting", page_icon="👥", layout="wide")

st.title("👥 HAWK-EYE — Real-Time Crowd Monitoring System")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("YOLO model", ["yolo11n.pt", "yolov8n.pt"])
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
    low_max = st.number_input("Low crowd: maximum people", min_value=0, max_value=500, value=10)
    moderate_max = st.number_input("Moderate crowd: maximum people", min_value=1, max_value=1000, value=25)

def crowd_level(count, low, moderate):
    if count <= low:
        return "LOW", "🟢"
    if count <= max(moderate, low):
        return "MODERATE", "🟠"
    return "HIGH", "🔴"

tab_video, tab_camera, tab_about = st.tabs(
    ["Video analysis", "Webcam", "About"]
)

with tab_video:
    st.subheader("Analyze a video")
    uploaded = st.file_uploader(
        "Upload an MP4, MOV, or AVI video",
        type=["mp4", "mov", "avi"],
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        frame_skip = st.select_slider(
            "Process every Nth frame",
            options=[1, 2, 3, 5, 8],
            value=1,
        )
    with col_b:
        max_width = st.select_slider(
            "Display width",
            options=[480, 640, 800, 960],
            value=800,
        )
    st.caption("ByteTrack estimates how many different people appear. Use N = 1 for best results.")

    run = st.button("▶ Start video analysis", type="primary", disabled=not uploaded)
    if run:
        temp_dir = tempfile.mkdtemp(prefix="crowd_video_")
        cap = None
        try:
            suffix = os.path.splitext(uploaded.name)[1]
            temp_path = os.path.join(temp_dir, f"uploaded{suffix}")
            with open(temp_path, "wb") as video_file:
                video_file.write(uploaded.getbuffer())

            cap_info = cv2.VideoCapture(temp_path)
            if not cap_info.isOpened():
                cap_info.release()
                st.error("Could not open this video. Try another supported video file.")
                st.stop()
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            video_fps = cap_info.get(cv2.CAP_PROP_FPS) or 25
            cap_info.release()
            st.info(
                f"Video length: approximately {total_frames / video_fps:.1f} seconds "
                f"| Source FPS: {video_fps:.1f}"
            )

            model = load_person_model(model_choice)
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("OpenCV could not read this video. Try another MP4, MOV, or AVI file.")
                st.stop()
            frame_slot = st.empty()
            progress = st.progress(0)
            status = st.empty()
            metric_cols = st.columns(4)
            count_metric = metric_cols[0].empty()
            level_metric = metric_cols[1].empty()
            fps_metric = metric_cols[2].empty()
            processed_metric = metric_cols[3].empty()
            chart_slot = st.empty()
            history = []
            seen_track_ids = set()
            tracking_ids_available = True
            frame_idx = 0
            processed = 0
            started = time.perf_counter()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % frame_skip != 0:
                    continue

                annotated, count, track_ids, has_track_ids, infer_time = track_people(
                    frame,
                    model,
                    conf,
                )
                seen_track_ids.update(track_ids)
                tracking_ids_available = tracking_ids_available and has_track_ids
                display_h = int(annotated.shape[0] * (max_width / annotated.shape[1]))
                display = cv2.resize(annotated, (max_width, display_h))
                level, emoji = crowd_level(count, int(low_max), int(moderate_max))
                elapsed = max(time.perf_counter() - started, 1e-9)
                processing_fps = processed / elapsed if processed else 0.0
                processed += 1
                history.append({"Frame": frame_idx, "People detected": count, "Crowd level": level})
                frame_slot.image(
                    cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                count_metric.metric("People in current frame", count)
                level_metric.metric("Crowd level", f"{emoji} {level}")
                fps_metric.metric("Processing FPS", f"{processing_fps:.1f}")
                processed_metric.metric("Frames processed", processed)
                status.write(f"Processing frame {frame_idx:,} of {total_frames:,}…")
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                if processed % 5 == 0:
                    hist_df = pd.DataFrame(history)
                    fig = px.line(
                        hist_df,
                        x="Frame",
                        y="People detected",
                        title="People detected over time",
                    )
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    chart_slot.plotly_chart(fig, use_container_width=True)

            if history:
                hist_df = pd.DataFrame(history)
                status.success("Analysis complete.")
                st.subheader("Summary statistics")
                s1, s2, s3 = st.columns(3)
                s1.metric("Peak detected count", int(hist_df["People detected"].max()))
                s2.metric("Average detected count", f'{hist_df["People detected"].mean():.1f}')
                s3.metric("Frames analyzed", len(hist_df))
                st.metric(
                    "Estimated unique people tracked",
                    len(seen_track_ids),
                    help=(
                        "Number of distinct ByteTrack IDs observed. Tracking can "
                        "still assign a new ID to the same person after a long "
                        "occlusion or missed detections."
                    ),
                )
                if not tracking_ids_available:
                    st.warning(
                        "Some detections did not receive a ByteTrack ID, so the "
                        "unique-person total may be incomplete."
                    )
                fig = px.line(hist_df, x="Frame", y="People detected", title="Final crowd-count timeline")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(hist_df.tail(20), use_container_width=True)
                st.download_button(
                    "⬇ Download count history (CSV)",
                    hist_df.to_csv(index=False).encode("utf-8"),
                    "crowd_count_history.csv",
                    "text/csv",
                )
            else:
                st.warning("No frames could be processed.")
        finally:
            if cap is not None:
                cap.release()
            shutil.rmtree(temp_dir)

with tab_camera:
    st.subheader("Webcam mode")
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0)
    camera_col1, camera_col2 = st.columns(2)
    with camera_col1:
        start_camera = st.button("📷 Start webcam", type="primary")
    with camera_col2:
        stop_camera = st.button("⏹ Stop webcam")
    if stop_camera:
        st.session_state["camera_running"] = False
    if start_camera:
        st.session_state["camera_running"] = True

    if st.session_state.get("camera_running", False):
        model = load_person_model(model_choice)
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("Could not open the webcam. Check camera permissions or try the Video analysis tab.")
            st.session_state["camera_running"] = False
        else:
            camera_frame = st.empty()
            c1, c2, c3 = st.columns(3)
            m_count, m_level, m_fps = c1.empty(), c2.empty(), c3.empty()
            chart = st.empty()
            history = deque(maxlen=150)
            seen_track_ids = set()
            tracking_ids_available = True
            m_unique = st.empty()
            tracking_warning = st.empty()
            st.info("Click Stop webcam to end the capture.")
            frame_num = 0
            while cap.isOpened() and st.session_state.get("camera_running", False):
                ok, frame = cap.read()
                if not ok:
                    st.warning("Camera frame could not be read.")
                    break
                annotated, count, track_ids, has_track_ids, infer_time = track_people(
                    frame,
                    model,
                    conf,
                )
                seen_track_ids.update(track_ids)
                tracking_ids_available = tracking_ids_available and has_track_ids
                level, emoji = crowd_level(count, int(low_max), int(moderate_max))
                frame_num += 1
                history.append({"Frame": frame_num, "People detected": count, "Crowd level": level})
                camera_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                m_count.metric("People in current frame", count)
                m_level.metric("Crowd level", f"{emoji} {level}")
                m_fps.metric("Inference FPS", f"{1 / max(infer_time, 1e-9):.1f}")
                m_unique.metric(
                    "Estimated unique people tracked",
                    len(seen_track_ids),
                )
                if not tracking_ids_available:
                    tracking_warning.warning(
                        "Some detections did not receive a ByteTrack ID, so the "
                        "unique-person total may be incomplete."
                    )
                if frame_num % 5 == 0:
                    fig = px.line(pd.DataFrame(history), x="Frame", y="People detected", title="Live count history")
                    fig.update_layout(height=280)
                    chart.plotly_chart(fig, use_container_width=True)
            cap.release()
            st.session_state["camera_running"] = False

with tab_about:
    st.subheader("About")
    st.markdown("""
    CrowdWatch detects people with YOLO and tracks them with ByteTrack.
    The unique-person count is an estimate; missed detections or occlusion
    can cause a person to receive a new tracking ID.
    """)
