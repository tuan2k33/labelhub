# LabelHub: design notes

LabelHub takes object-detection data from raw images to a trained, versioned model on one small GPU (4 GB): label boxes by hand or with SAM3, freeze versions of images, labels and models, train YOLO, and try the result. Scope is bounding boxes in YOLO format for a single user. Segmentation masks and video are out of scope for now.

This file records what was built and why. Day-to-day instructions for working in the code are in `CLAUDE.md`, and the user-facing overview is in `README.md`.

## Principles

- Nothing is overwritten: images are content-addressed and every version is immutable, so any model can be traced back to the exact data it saw.
- It must run on a 4 GB GPU. Heavy steps are sequential, and training can also happen elsewhere (Colab, Kaggle) and be imported back.
- Users never touch SQL. SQLite through SQLModel, one database per dataset.
- YOLO format only. Converters (`yolo2coco`, `coco2yolo`) would be separate scripts, not part of the app.

## Datasets

A dataset is an independent folder, `<data root>/projects/<slug>/`, holding its own database, images, models and thumbnails. Classes, versions, models and auto-labelling all belong to one dataset, so a set of forklift photos never mixes with a set of street signs. The UI has a Datasets page and a switcher in the top bar. A database sitting directly in the data root (the layout from before datasets existed) shows up as the dataset `default`.

All datasets share one loaded SAM3 model and one GPU lock, so only one GPU job runs at a time across the whole server.

## Three kinds of version

Three independent lines of versions, each named `YYYY.MM.x` (`x` counts up within a month and restarts at 1 the next month, e.g. `2026.10.3`). Each version has a parent, a note and a diff against another version.

- **ImageVersion**: a frozen set of images plus its split into train, val, test and gold. The split is derived from the image hash and a seed, so an image keeps its split when the dataset grows.
- **LabelVersion**: a frozen snapshot of the accepted boxes plus the class list, with a count of where the boxes came from (human, `auto:sam3`). It is keyed by image hash, so it is not tied to one ImageVersion. By default it only covers images that were reviewed; Auto mode also includes images SAM3 has labelled but nobody opened.
- **ModelVersion**: the weights, the metrics, the config, and the exact pair (ImageVersion, LabelVersion) it was trained on, plus a parent when it was fine-tuned from another model.

The **workspace** is the live, freely editable state of images and boxes, and is not itself a version. A box has a status: `accepted` (in use), `rejected` (an automatic box below the confidence the user chose: hidden but kept, so lowering the threshold brings it back), or `proposed` (only produced by the older flow; opening the image settles it into accepted or rejected).

**Training data is ImageVersion ∩ LabelVersion**: images in the image version that also have labels in the label version. An image that was reviewed and has no boxes is a valid background image; an image that was never reviewed is left out. A per-image `reviewed` flag tells the two apart. Gold images are excluded from training.

The UI shows the link in both directions: from a model to its data and label sources, and from a version to the models trained on it.

## Storage

Images are stored by SHA-256, so importing the same file twice stores it once. A version is only a small manifest of hashes, which makes creating one cheap. The split ratios and seed are stored with the ImageVersion. Creating a version is refused when nothing changed since the previous one.

## Working modes

**Manual.** SAM3 draws boxes for the class names, and boxes at or above 0.5 become labels straight away. The person then opens images one at a time. Opening an image counts as reviewing it: there is no review button and no per-box accept or reject. Wrong boxes are deleted, missing ones drawn, wrong classes changed. Each box has its own eye toggle (the `H` key shows or hides all of them), which only affects what is drawn. A confidence slider from 0.3 to 1.0 in the Annotations panel decides which automatic boxes stay: dragging it up turns lower boxes into `rejected`, dragging it down brings them back, so the labels in a version always match what is on screen. Then the person picks an ImageVersion and a LabelVersion on the Models page and trains.

**Auto.** One button runs the whole chain without a review step: label the new images with SAM3 (boxes at 0.5 and above), freeze an ImageVersion and a LabelVersion (including unopened images), train YOLO, and score the model. SAM3 is unloaded from the GPU before training starts. The run can be cancelled, and the result can be inspected and fixed afterwards in Images. The state of a run lives in server memory, so a server restart mid-run loses the step display while the training process finishes on its own.

**Gold images.** A gold image is one the user has opened, fixed and ticked "Gold image" (only possible after opening it). Gold images get their own split, are never trained on, and every model is scored on them separately. So each model reports two numbers: mAP50 against the labels of its own test split, and mAP50 against the gold images. This matters because a student trained on SAM3's labels and scored on SAM3's labels only measures how well it copies SAM3. Without gold images the UI says so.

## Auto-labelling

Backends sit behind one interface, `detect(image, classes) -> boxes`.

- **SAM3 (`facebook/sam3`, fp16) is the backend.** Measured on an RTX 3050 Ti Laptop with an 800x534 image: peak VRAM 2.1 GB, 0.58 s to encode an image once, about 0.13 s for each extra class. The vision features are computed once per image and reused for every class prompt. It needs a Hugging Face token and approved access to the gated model.
- **Thresholds.** The detector returns boxes from 0.3 upwards. That floor is the `Sam3Detector(threshold=0.3)` default in this code, which matches the `transformers` default. Auto-label accepts boxes at 0.5 and above; the slider in the editor can move the cut anywhere from 0.3 to 1.0.
- **GPU discipline.** SAM3 and YOLO training do not fit together in 4 GB, so SAM3 is released before any training starts, training runs in its own process, and inference falls back to the CPU while a model is training.

Considered and not used:

- EfficientSAM3: lighter (1.2 GB peak) but much less confident (0.35 to 0.65 against 0.84 to 0.96), and full SAM3 already fits.
- SAM 3.1 (Object Multiplex): the speed-up is for tracking many objects in video, so it does not help still images.
- Free vision-language models through OpenRouter: daily quotas are too small for a dataset of a thousand images, and box coordinates from a VLM would need checking.
- Exporting SAM3 to TensorRT with a smaller input size for faster labelling, using the export scripts in the author's SAM3-DeepStream repository: promising for speed, but labelling 1,000 images in about 19 minutes was judged good enough.

## Training

- YOLO26 n or s, 640 px, batch 8 by default, AMP on. It starts from pretrained weights or from one of the user's own models (fine-tuning).
- A run is a separate process (`labelhub.train_worker`) in its own process group. The ModelVersion row in the database is the source of truth: progress, metrics and errors are written there, cancelling sends SIGTERM to the group, and a worker that disappeared is marked failed. There is no queue, just one job at a time.
- **Training elsewhere.** The export bundle holds the YOLO dataset and a `train.py`; after training, `best.pt` is uploaded back and registered against the same pair of versions.
- Metrics: mAP50, mAP50-95, precision, recall, mAP50 per class, and the gold scores when gold images exist.

## Inference

- Pick a trained or imported model and an image (drop a file, choose one from the dataset, or take a random one). The model runs immediately and the boxes are drawn over the image. The server always returns detections from 0.05 up, and the confidence slider filters in the browser. Image size can be overridden; by default it is the size the model was trained at, or 640 for imported models that carry no size.
- Two models can be compared on the same image, side by side with a table of boxes per class. For dataset images, the stored labels can be overlaid as white dashed boxes.
- The same endpoint is public: `POST /api/p/<dataset>/models/<model>/predict` takes a file or an image hash plus `conf` and `imgsz`, and returns boxes, `total_ms`, `inference_ms`, the device and a `cold` flag. The UI offers curl and Python snippets.
- Every run is logged. Median, 95th percentile and mean latency per model appear on the model card, leaving out the first run after loading because it includes warm-up.

## Architecture

- **Backend**: FastAPI, SQLModel on SQLite. A registry mounts one API per dataset under `/api/p/<slug>` and selects the dataset's folder per request with a context variable. Shared between datasets: the loaded SAM3 model, the GPU lock, the Auto lock, and the job table.
- **Jobs**: auto-label runs in a thread of the server with progress polling; the Auto pipeline runs in a thread; training runs as a subprocess.
- **Frontend**: React, Vite and TypeScript, with a Konva canvas for boxes. The dataset is chosen by the URL hash. A static build with a mock API (`npm run build:demo`) runs without a server and is used for the Claude artifact demo.
- One user, no login.

## Status

All planned phases are done: media by hash, versions, split and YOLO export; image and label versions; the UI with multiple datasets; SAM3 auto-label and the Manual flow; training, model versions and the Colab bundle; Auto mode and gold images; inference and model comparison.

Not planned unless needed: a TensorRT backend for SAM3.

Open ideas:

- A self-training loop: a trained YOLO proposes boxes for new images (faster than SAM3, source `auto:yolo:<model version>`), the person fixes them, and the model is retrained.
- DINOv3 features to find near-duplicate images or suggest classes. It cannot produce boxes from text.
- Video: the unit to label would be frames extracted by some rule that is still to be decided.
- A class mapping between two label versions, which is only needed for diffs or fine-tuning across different class lists.

## References

- SAM 3: https://huggingface.co/facebook/sam3 (gated, request access) and https://github.com/facebookresearch/sam3
- SAM 3.1, Object Multiplex: https://ai.meta.com/blog/segment-anything-model-3/
- SAM3-DeepStream, the author's TensorRT export scripts: https://github.com/tuan2k33/SAM3-DeepStream
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- Labelling tools that shaped the UI, compared in vendor write-ups (read with care): https://www.cvat.ai/blog/cvat-or-label-studio-which-one-to-choose and https://roboflow.com/compare-labeling-tools/cvat-vs-label-studio
- Linking data versions to models, in the style of DVC plus MLflow: https://aws.amazon.com/blogs/machine-learning/end-to-end-lineage-with-dvc-and-amazon-sagemaker-ai-mlflow-apps/
