# Plant Intelligence Demo

A two-tab Streamlit application built for a cement manufacturing client demo, showcasing two
AI agent use cases:

1. **VisionAI — Plant Safety**: YOLOv8-based PPE (personal protective equipment) compliance
   detection on uploaded images and short videos.
2. **HRMS Multi-Agent Assistant**: A 3-agent LangGraph chain (Intake, Resolver, Escalation)
   answering employee HR queries via a hosted LLM provider selectable at runtime (Groq or
   OpenAI), with visible human-escalation guardrails.

## Project structure

```
app.py                          Main entry point, tab routing, shared CSS
tabs/vision_ppe.py               Tab 1: PPE detection logic
tabs/hrms_agents.py              Tab 2: HRMS assistant UI
agents/graph.py                  LangGraph 3-agent chain definition
data/mock_leave_balances.csv     Sample fictional employee leave data
data/sample_hr_policy.txt        Sample HR policy document (leave / WFH / reimbursement)
models/                          Folder for YOLOv8 PPE weight file (see models/README.md)
requirements.txt
packages.txt                     System (apt) packages required by opencv on Streamlit Cloud
.streamlit/secrets.toml.example  Template for the Groq API key
```

## Setup (local)

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key(s):

   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

   Edit `.streamlit/secrets.toml` and replace the placeholder(s) with your real key(s):

   ```toml
   GROQ_API_KEY = "your-actual-groq-key"
   OPENAI_API_KEY = "your-actual-openai-key"
   ```

   You only need to set the key for whichever provider(s) you plan to demo — the app's
   "Model provider settings" panel also lets you paste a key directly into the UI at
   runtime (useful for testing a different key live without touching this file); a
   UI-entered key overrides secrets.toml for that session only and is never persisted.
   Get a Groq key from https://console.groq.com/keys and an OpenAI key from
   https://platform.openai.com/api-keys. `.streamlit/secrets.toml` is your local secret
   file and should never be committed to version control.

4. (Optional but recommended) Add PPE detection weights. See **PPE weights** below — without
   this file, Tab 1 runs in a graceful fallback mode (generic "person" detection only, using
   the base `yolov8n.pt` model, which is downloaded automatically on first run).

5. Run the app:

   ```bash
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501`.

## PPE weights — action required

This build does **not** bundle a fine-tuned PPE-detection weight file, since redistributing
third-party trained weights is outside the scope of this build. To get full PPE classification
(helmet / no-helmet / vest / no-vest, etc.) instead of the generic-person fallback, download a
pretrained, Ultralytics-compatible PPE YOLOv8 weight file and place it at:

```
models/ppe_yolov8.pt
```

Recommended sources (see `models/README.md` for full detail):

- Hugging Face: `keremberke/yolov8m-hard-hat-detection` (or similar PPE-detection models by
  the same author) — https://huggingface.co/keremberke/yolov8m-hard-hat-detection
- Roboflow Universe — search "PPE detection" or "hard hat detection" for a dataset with a
  trained YOLOv8 checkpoint export.
- Train your own with `ultralytics` on an open PPE dataset.

Until this file is added, the app runs correctly but only detects the generic "person" class,
with a visible on-screen notice explaining why.

## Deployment on Streamlit Cloud

1. Push this project to a GitHub repository (do **not** commit `.streamlit/secrets.toml` —
   only commit `.streamlit/secrets.toml.example`). Make sure `packages.txt` is committed too —
   it installs the system graphics libraries OpenCV needs on Streamlit Cloud's minimal
   container (see Troubleshooting below if you skip this and hit an `ImportError` on `cv2`).
2. In Streamlit Cloud, create a new app pointing at this repository and `app.py`.
3. In the app's **Settings > Secrets**, paste the contents of your local
   `.streamlit/secrets.toml` (with your real `GROQ_API_KEY`).
4. Deploy. First load will download the base YOLOv8 weights (`yolov8n.pt`) automatically if
   no PPE-specific weights are bundled in the repository.
5. If you want full PPE classification in the deployed app, commit `models/ppe_yolov8.pt` to
   the repository (check file size limits — Streamlit Cloud free tier and GitHub both have
   repository size constraints; use Git LFS if the weight file is large).

## Notes on design choices

- **Video processing** is frame-sampled (configurable "process every Nth frame" and a max
  sampled-frame cap in the UI) rather than full-framerate, so it stays responsive on
  Streamlit Cloud's free-tier CPU. Annotated sampled frames are shown as a frame gallery
  (large selectable view plus a thumbnail grid) rather than reassembled into an output video,
  since typical OpenCV builds lack an H.264 encoder and browsers often cannot play the
  resulting file inline.
- **HR policy retrieval** uses a lightweight in-memory TF-IDF retriever (`scikit-learn`) over
  paragraph-level chunks of `data/sample_hr_policy.txt`, avoiding a heavier vector-database or
  embedding-model dependency while remaining accurate for this demo's scope.
- **Model resource caching**: both the YOLO model (`tabs/vision_ppe.py`) and the compiled
  LangGraph agent chain (`tabs/hrms_agents.py`) are wrapped in `@st.cache_resource`, so they
  are loaded/compiled once per app instance rather than on every rerun.
- **Escalation guardrail**: the Escalation Agent never attempts to answer grievance,
  disciplinary, or ambiguous queries — it only drafts a routing note and marks the query as
  escalated, by design, so the demo clearly shows human-in-the-loop behavior.

## Troubleshooting

- **"API key is not configured"**: either paste a key into the "Model provider settings"
  panel in the app, or ensure `.streamlit/secrets.toml` exists locally (or the secret is set
  in Streamlit Cloud) with a valid `GROQ_API_KEY` / `OPENAI_API_KEY`.
- **Rate limit / auth errors**: the app catches these and shows a plain-language message
  instead of a stack trace; wait and retry, or verify the key for the selected provider.
- **"Model not found" / model rejected errors**: hosted providers periodically retire model
  IDs. Check `agents/graph.py`'s `PROVIDER_CONFIG` against the provider's current model list
  (e.g. https://console.groq.com/docs/models for Groq) and update the IDs there.
- **Video upload feels slow**: reduce "max sampled frames" or increase "process every Nth
  frame" in the Tab 1 controls.
- **`ImportError` at `import cv2` (works locally, fails on Streamlit Cloud)**: `ultralytics`
  depends on plain `opencv-python` (GUI-enabled), which conflicts with the
  `opencv-python-headless` pinned in `requirements.txt` — both install the same `cv2.abi3.so`
  file, and whichever wins may need system graphics libraries (`libGL.so.1`, etc.) that a
  minimal cloud container doesn't have by default, even though your local desktop already has
  them installed as part of its normal graphics stack. This repo includes a `packages.txt` at
  the root that tells Streamlit Cloud to `apt-get install` the missing libraries
  (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`) — make sure it's committed,
  then use "Reboot app" in Streamlit Cloud's app menu (a plain rerun won't re-read
  `packages.txt`).
