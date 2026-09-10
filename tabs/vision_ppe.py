"""
Tab 1: VisionAI - Plant Safety (PPE Detection).

Runs a YOLOv8 model over an uploaded image or short video to detect PPE compliance
(helmet, safety vest, no-PPE), overlays bounding boxes, and shows a Detected Events
table with a compliance alert summary.

If no fine-tuned PPE weight file is present under models/ppe_yolov8.pt, the app
falls back to the base yolov8n.pt model restricted to the "person" class only, and
displays a clear notice explaining that PPE-specific classes require the
fine-tuned weights to be added.
"""

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PPE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "ppe_yolov8.pt")
FALLBACK_WEIGHTS_NAME = "yolov8n.pt"

# BGR color tuples (cv2 native order), matched to the app's navy accent theme.
STATUS_COLORS_BGR = {
    "compliant": (66, 133, 33),     # muted green
    "violation": (40, 39, 196),     # muted red
    "neutral": (163, 92, 15),       # navy-ish blue for informational-only detections
}

STATUS_LABELS = {
    "compliant": "Compliant",
    "violation": "Violation",
    "neutral": "Detected",
}


def classify_ppe_status(class_name: str) -> str:
    """Maps a detector class name to compliant / violation / neutral.

    Written to be tolerant of common PPE dataset naming conventions
    (e.g. "Hardhat" / "NO-Hardhat" / "Safety Vest" / "NO-Safety Vest" / "Mask").
    Adjust here if a bundled weight file uses different class names.
    """
    name = class_name.strip().lower()

    if name in ("person",):
        return "neutral"

    negation_markers = ("no-", "no_", "no ", "without", "missing")
    if any(name.startswith(m) or f" {m}" in f" {name}" for m in negation_markers):
        return "violation"

    ppe_keywords = ("helmet", "hardhat", "hard hat", "vest", "mask", "glove", "goggle", "ppe")
    if any(k in name for k in ppe_keywords):
        return "compliant"

    return "neutral"


@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    """Loads the PPE-specific YOLOv8 model if bundled, otherwise falls back to
    the base yolov8n.pt model. Returns (model, ppe_mode: bool)."""
    from ultralytics import YOLO

    if os.path.exists(PPE_WEIGHTS_PATH):
        model = YOLO(PPE_WEIGHTS_PATH)
        return model, True

    model = YOLO(FALLBACK_WEIGHTS_NAME)
    return model, False


def _draw_box(frame_bgr: np.ndarray, xyxy, label: str, color) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y1 = max(y1 - text_h - 8, 0)
    cv2.rectangle(frame_bgr, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 3, y1 - 5 if y1 - 5 > 0 else label_y1 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_detection_on_frame(model, ppe_mode: bool, frame_bgr: np.ndarray, conf_threshold: float):
    """Runs YOLO on a single BGR frame, draws annotations in place on a copy,
    and returns (annotated_frame_bgr, events)."""
    results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)
    result = results[0]
    annotated = frame_bgr.copy()
    events = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, events

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]

        if not ppe_mode and cls_name.lower() != "person":
            continue

        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        status = classify_ppe_status(cls_name)
        color = STATUS_COLORS_BGR[status]

        label_text = f"{cls_name} {confidence:.2f}"
        _draw_box(annotated, xyxy, label_text, color)

        events.append(
            {
                "class": cls_name,
                "confidence": round(confidence, 3),
                "status": STATUS_LABELS[status],
                "_status_key": status,
            }
        )

    return annotated, events


def process_image(model, ppe_mode: bool, pil_image: Image.Image, conf_threshold: float):
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    annotated_bgr, events = run_detection_on_frame(model, ppe_mode, bgr_array, conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    for e in events:
        e["frame"] = "Image"
        e["timestamp"] = "-"

    return annotated_rgb, events


def process_video(model, ppe_mode: bool, video_path: str, conf_threshold: float, sample_every_n: int, max_sampled_frames: int):
    """Runs detection on sampled frames and returns annotated frames as images
    (not a re-encoded video). Standard OpenCV builds used on Streamlit Cloud and
    in most Python environments do not ship an H.264 encoder, so re-encoding
    annotated frames into an .mp4 frequently produces a file browsers cannot
    play inline — the bounding boxes would then be invisible even though
    detection ran correctly. Returning a gallery of full-resolution annotated
    frames sidesteps that codec issue entirely and is guaranteed to render."""
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    sampled_count = 0
    all_events = []
    sampled_frames = []  # list of {"frame": int, "timestamp": str, "image": RGB ndarray}

    progress_bar = st.progress(0.0, text="Processing sampled frames...")
    total_frames_hint = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    expected_samples = max(1, min(max_sampled_frames, int(total_frames_hint // sample_every_n) + 1))

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            annotated_bgr, events = run_detection_on_frame(model, ppe_mode, frame_bgr, conf_threshold)
            timestamp_sec = frame_idx / source_fps
            timestamp_label = f"{timestamp_sec:.2f}s"

            for e in events:
                e["frame"] = frame_idx
                e["timestamp"] = timestamp_label

            all_events.extend(events)
            sampled_frames.append(
                {
                    "frame": frame_idx,
                    "timestamp": timestamp_label,
                    "image": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    "has_detections": len(events) > 0,
                }
            )
            sampled_count += 1

            progress_bar.progress(
                min(sampled_count / expected_samples, 1.0),
                text=f"Processed {sampled_count} sampled frame(s)...",
            )

            if sampled_count >= max_sampled_frames:
                break

        frame_idx += 1

    cap.release()
    progress_bar.empty()

    return sampled_frames, all_events, sampled_count


def _render_alert_panel(events: list):
    if not events:
        st.info("No detections above the confidence threshold were found.")
        return

    violation_count = sum(1 for e in events if e["_status_key"] == "violation")
    compliant_count = sum(1 for e in events if e["_status_key"] == "compliant")
    neutral_count = sum(1 for e in events if e["_status_key"] == "neutral")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Compliant detections", compliant_count)
    with cols[1]:
        st.metric("Violations", violation_count)
    with cols[2]:
        st.metric("Other detections", neutral_count)

    if violation_count > 0:
        st.error(f"{violation_count} PPE violation(s) detected. Review the events table below.")
    elif compliant_count > 0:
        st.success("No PPE violations detected in the processed content.")


def render():
    st.markdown("### VisionAI — Plant Safety")
    st.write(
        "Upload an image or a short video from the plant floor to detect PPE "
        "compliance. Detected objects are shown with bounding boxes and "
        "confidence scores, with a summary of compliant and violation events."
    )

    model, ppe_mode = load_model()

    if not ppe_mode:
        st.warning(
            "PPE-specific weights were not found at models/ppe_yolov8.pt. Running the "
            "base YOLOv8 model restricted to \"person\" detection only. To enable full "
            "PPE classification (helmet, vest, no-PPE), add a fine-tuned PPE weight "
            "file to the models/ folder — see models/README.md for sources."
        )

    col_left, col_right = st.columns([2, 1])
    with col_right:
        conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        sample_every_n = st.slider("Video: process every Nth frame", min_value=5, max_value=60, value=15, step=5)
        max_sampled_frames = st.slider("Video: max sampled frames", min_value=10, max_value=80, value=30, step=10)

    with col_left:
        media_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

        if media_type == "Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.file_uploader("Upload a short video", type=["mp4"])

    if uploaded_file is None:
        st.info("Upload a file to run detection.")
        return

    if media_type == "Image":
        pil_image = Image.open(uploaded_file)
        with st.spinner("Running detection..."):
            annotated_rgb, events = process_image(model, ppe_mode, pil_image, conf_threshold)

        st.markdown("#### Detection Output")
        st.image(annotated_rgb, width="stretch")

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_video_path = tmp_file.name

        with st.spinner("Running detection on sampled frames..."):
            sampled_frames, events, sampled_count = process_video(
                model, ppe_mode, tmp_video_path, conf_threshold, sample_every_n, max_sampled_frames
            )

        os.unlink(tmp_video_path)

        st.markdown("#### Detection Output")
        st.caption(
            f"Processed {sampled_count} sampled frame(s) (every {sample_every_n}th frame, "
            f"capped at {max_sampled_frames}) for responsiveness on limited CPU resources. "
            f"Each sampled frame is shown as an annotated snapshot below rather than a "
            f"reassembled video, so bounding boxes render reliably in every browser."
        )

        if sampled_frames:
            frame_labels = [
                f"Frame {f['frame']} ({f['timestamp']})" + (" — detection" if f["has_detections"] else "")
                for f in sampled_frames
            ]
            default_index = next((i for i, f in enumerate(sampled_frames) if f["has_detections"]), 0)
            selected_label = st.select_slider(
                "Sampled frame",
                options=frame_labels,
                value=frame_labels[default_index],
            )
            selected_index = frame_labels.index(selected_label)
            st.image(sampled_frames[selected_index]["image"], width="stretch")

            with st.expander(f"View all {len(sampled_frames)} processed frames", expanded=False):
                cols_per_row = 3
                for row_start in range(0, len(sampled_frames), cols_per_row):
                    row_frames = sampled_frames[row_start:row_start + cols_per_row]
                    row_cols = st.columns(cols_per_row)
                    for col, f in zip(row_cols, row_frames):
                        with col:
                            st.image(
                                f["image"],
                                caption=f"Frame {f['frame']} ({f['timestamp']})",
                                width="stretch",
                            )
        else:
            st.info("No frames could be processed from the uploaded video.")

    st.markdown("#### Alert Summary")
    _render_alert_panel(events)

    st.markdown("#### Detected Events")
    if events:
        df = pd.DataFrame(events)[["frame", "timestamp", "class", "confidence", "status"]]
        df = df.rename(
            columns={
                "frame": "Frame",
                "timestamp": "Timestamp",
                "class": "Class",
                "confidence": "Confidence",
                "status": "Status",
            }
        )
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.write("No events to display.")
