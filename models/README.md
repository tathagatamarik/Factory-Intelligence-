# Models folder

This folder is where the fine-tuned PPE-detection YOLOv8 weight file should be placed.

## Required file

Place a PPE-detection weight file here named:

    models/ppe_yolov8.pt

The application looks for this exact filename first. If it is not present, the app falls back
automatically to the base `yolov8n.pt` model (downloaded on first run by the `ultralytics`
package) and restricts detection to the generic "person" class only, with an on-screen notice
explaining that PPE-specific classes (helmet, vest, no-helmet, no-vest, etc.) require the
fine-tuned weights below.

## Where to get pretrained PPE weights

You (the client) need to source and download one of the following community-trained,
Ultralytics-compatible PPE detection weight files yourself, since redistributing third-party
model weights is outside the scope of this build. Recommended sources:

1. **Keremberke PPE detection model (Hugging Face)**
   `keremberke/yolov8m-hard-hat-detection` (or the "PPE" variant by the same author)
   https://huggingface.co/keremberke/yolov8m-hard-hat-detection
   Classes typically include: `helmet`, `no-helmet` (model-specific — check its model card).

2. **Roboflow Universe "PPE Detection" / "Hard Hat Workers" datasets and trained checkpoints**
   https://universe.roboflow.com/  — search "PPE detection" or "hard hat detection".
   Roboflow lets you export a trained YOLOv8 checkpoint (`best.pt`) directly, or train your own
   in a few clicks using their hosted training if you only have a labeled dataset.

3. **Train your own** using the open "Hard Hat Workers Dataset" or "CHVG PPE dataset" with
   `ultralytics` (`yolo train data=ppe.yaml model=yolov8n.pt`), then copy the resulting
   `runs/detect/train/weights/best.pt` here and rename it to `ppe_yolov8.pt`.

## Expected class names

The app's post-processing logic in `tabs/vision_ppe.py` assumes class names similar to:

    helmet, no-helmet, vest, no-vest, person

If your chosen weight file uses different class names, adjust the `PPE_CLASS_MAP` dictionary
at the top of `tabs/vision_ppe.py` to map your model's actual class names to the
compliant / violation categories used by the alert panel.

## License note

Always check the license of any third-party weight file before using it in a client-facing
or commercial demo.
