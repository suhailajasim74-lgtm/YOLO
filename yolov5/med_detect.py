# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference with MediaPipe Hand Landmark pipeline.

The pipeline works in two stages:
  Stage 1 – YOLOv5 detects and crops the hand region from the frame.
  Stage 2 – MediaPipe extracts 21 3D landmarks from the crop, which are
             then classified by a lightweight MLP (trained separately or
             auto-trained on first run if a landmarks CSV exists).

Usage:
    # Webcam (recommended for sign language)
    $ python detect.py --weights best.pt --source 0 --use-mediapipe

    # Image / video (same flags as before)
    $ python detect.py --weights best.pt --source img.jpg --use-mediapipe

    # Disable MediaPipe and run classic YOLOv5 only
    $ python detect.py --weights best.pt --source 0

Landmark CSV format (for training the MLP classifier):
    Each row: label, x0, y0, z0, x1, y1, z1, ... x20, y20, z20  (63 values after label)
    Save such a CSV with --save-landmarks during a collection run.
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe Landmark Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_mediapipe():
    """
    Import mediapipe lazily so the rest of detect.py works without it.
    Returns the mediapipe module (0.10+ Tasks API).
    """
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        raise ImportError(
            "MediaPipe is not installed. Run:  pip install mediapipe"
        )


def _get_model_path(model_path: str = "") -> str:
    """
    Resolve the hand_landmarker.task model file path.
    If not provided or not found, download it automatically from Google.
    The model is cached next to this script as 'hand_landmarker.task'.
    """
    import urllib.request

    default_local = Path(__file__).parent / "hand_landmarker.task"
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

    candidate = Path(model_path) if model_path else default_local

    if candidate.exists():
        return str(candidate)

    LOGGER.info(f"[MediaPipe] Downloading hand_landmarker.task model to {default_local} …")
    urllib.request.urlretrieve(url, str(default_local))
    LOGGER.info("[MediaPipe] Download complete.")
    return str(default_local)


def build_hand_detector(mp, max_num_hands=1, min_detection_confidence=0.5,
                        model_path: str = ""):
    """
    Build a MediaPipe HandLandmarker using the new Tasks API (mediapipe 0.10+).
    Falls back gracefully if the model file cannot be downloaded.
    """
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    resolved_model = _get_model_path(model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model),
        running_mode=RunningMode.IMAGE,          # IMAGE mode: one frame at a time
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_landmarks(hand_detector, bgr_crop):
    """
    Run MediaPipe HandLandmarker (Tasks API 0.10+) on a BGR crop.
    Returns flat numpy array of 63 values (21 x [x,y,z]), normalised.
    Returns None if no hand found.
    """
    import mediapipe as mp_inner

    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp_inner.Image(image_format=mp_inner.ImageFormat.SRGB, data=rgb)
    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)

    # ── Normalisation ──────────────────────────────────────────────────────
    # Translate so that wrist (landmark 0) is the origin, then scale by the
    # distance between wrist and middle-finger MCP (landmark 9).
    # This makes the feature vector scale- and position-invariant, so the
    # same sign looks the same regardless of how far the hand is from the
    # camera or where it sits in the frame.
    wrist = coords[0].copy()
    coords -= wrist                          # translate to origin

    scale = np.linalg.norm(coords[9])       # dist wrist → middle MCP
    if scale > 1e-6:
        coords /= scale                      # normalise scale

    return coords.flatten()                  # shape (63,)


# Hand skeleton connections (21 MediaPipe landmarks)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index finger
    (0,9),(9,10),(10,11),(11,12),     # middle finger
    (0,13),(13,14),(14,15),(15,16),   # ring finger
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm cross-connections
]

def draw_landmarks_on_frame(mp_module, frame, hand_detector, bbox_xyxy):
    """
    Draw MediaPipe hand skeleton using pure OpenCV — no proto imports needed.
    Works with mediapipe 0.10+ Tasks API. Returns frame with skeleton drawn.
    """
    import mediapipe as mp_inner

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame

    ch, cw = crop.shape[:2]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp_inner.Image(image_format=mp_inner.ImageFormat.SRGB, data=rgb_crop)
    result = hand_detector.detect(mp_image)

    if result.hand_landmarks:
        crop_draw = crop.copy()
        for hand_lm in result.hand_landmarks:
            # Convert normalised [0,1] coords to pixel coords within the crop
            pts = [
                (int(lm.x * cw), int(lm.y * ch))
                for lm in hand_lm
            ]
            # Draw connections (bones)
            for start, end in _HAND_CONNECTIONS:
                cv2.line(crop_draw, pts[start], pts[end], (0, 200, 50), 2, cv2.LINE_AA)
            # Draw landmark dots
            for i, pt in enumerate(pts):
                # Fingertips (4,8,12,16,20) slightly larger, bright green
                is_tip = i in (4, 8, 12, 16, 20)
                color  = (0, 255, 0) if is_tip else (255, 255, 255)
                radius = 5 if is_tip else 3
                cv2.circle(crop_draw, pt, radius, (0, 0, 0), -1)          # black fill
                cv2.circle(crop_draw, pt, radius - 1, color, -1)          # coloured centre
        frame[y1:y2, x1:x2] = crop_draw

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Landmark MLP Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkClassifier:
    """
    A tiny sklearn MLP that classifies the 63-dim landmark vector.

    Training
    --------
    Either call .train(X, y) explicitly, or point it at a CSV file:
        clf = LandmarkClassifier.from_csv("landmarks.csv", label_names)
    The CSV must have columns: [label_index, x0, y0, z0, ..., x20, y20, z20]

    Inference
    ---------
        label_idx, confidence = clf.predict(landmark_vector)   # vector shape (63,)
    """

    def __init__(self, label_names: list):
        self.label_names = label_names
        self.model = None
        self.scaler = None

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])
        pipe.fit(X, y)
        self.model = pipe
        LOGGER.info(f"[LandmarkClassifier] Trained on {len(X)} samples, "
                    f"{len(self.label_names)} classes.")

    @classmethod
    def from_csv(cls, csv_path: str, label_names: list):
        """Build and train from a landmarks CSV file."""
        data = np.loadtxt(csv_path, delimiter=",")
        y = data[:, 0].astype(int)
        X = data[:, 1:].astype(np.float32)
        obj = cls(label_names)
        obj.train(X, y)
        return obj

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "labels": self.label_names}, f)
        LOGGER.info(f"[LandmarkClassifier] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["labels"])
        obj.model = d["model"]
        LOGGER.info(f"[LandmarkClassifier] Loaded from {path} "
                    f"({len(d['labels'])} classes)")
        return obj

    # ── Inference ─────────────────────────────────────────────────────────
    def predict(self, landmark_vec: np.ndarray):
        """Return (label_index, confidence_float) or (None, 0.0) on failure."""
        if self.model is None:
            return None, 0.0
        vec = landmark_vec.reshape(1, -1)
        proba = self.model.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        return idx, float(proba[idx])

    def is_ready(self):
        return self.model is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Main run() – extended with MediaPipe pipeline
# ─────────────────────────────────────────────────────────────────────────────

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=Path("C:/Users/suhai/Desktop/YOLO/data.yaml"),
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    # ── NEW MediaPipe args ────────────────────────────────────────────────
    use_mediapipe=False,          # enable the landmark pipeline
    landmark_clf_path="",         # path to a saved LandmarkClassifier .pkl
    landmark_csv_path="",         # path to a landmarks CSV for auto-training
    save_landmarks=False,         # save extracted landmarks to CSV
    mp_conf=0.5,                  # MediaPipe detection confidence
    mp_model_path="",              # path to hand_landmarker.task (auto-downloaded if empty)
    show_skeleton=True,           # draw MP skeleton on detections
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ── Load YOLOv5 model ────────────────────────────────────────────────
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    # ── Debug / YAML class-name sync (your original block) ───────────────
    print(f"\n{'='*60}")
    print("🔍 Model & Class Names")
    print(f"{'='*60}")
    try:
        import yaml
        yaml_path = Path(data)
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
            yaml_names = yaml_data.get("names", [])
            if isinstance(yaml_names, dict):
                yaml_names_list = [yaml_names[i] for i in sorted(yaml_names.keys())]
            else:
                yaml_names_list = list(yaml_names)
            if names != yaml_names_list:
                LOGGER.warning("⚠️  Class name mismatch – overriding with YAML names.")
                model.names = {i: n for i, n in enumerate(yaml_names_list)} \
                    if isinstance(model.names, dict) else yaml_names_list
                names = model.names
            print(f"✅ Classes ({len(names)}): {names}")
        else:
            print(f"❌ YAML not found: {yaml_path}")
    except Exception as e:
        print(f"❌ YAML load error: {e}")
    print(f"{'='*60}\n")

    imgsz = check_img_size(imgsz, s=stride)

    # ── MediaPipe setup ──────────────────────────────────────────────────
    mp_module      = None
    hand_detector  = None
    lm_classifier  = None
    lm_csv_writer  = None
    lm_csv_file    = None

    if use_mediapipe:
        LOGGER.info("[MediaPipe] Initialising hand landmark pipeline …")
        mp_module     = load_mediapipe()
        hand_detector = build_hand_detector(mp_module, min_detection_confidence=mp_conf, model_path=mp_model_path)

        # Resolve class names to a plain list for the classifier
        if isinstance(names, dict):
            label_names = [names[i] for i in sorted(names.keys())]
        else:
            label_names = list(names)

        # Load or train classifier
        if landmark_clf_path and Path(landmark_clf_path).exists():
            lm_classifier = LandmarkClassifier.load(landmark_clf_path)
        elif landmark_csv_path and Path(landmark_csv_path).exists():
            LOGGER.info(f"[MediaPipe] Training classifier from {landmark_csv_path} …")
            lm_classifier = LandmarkClassifier.from_csv(landmark_csv_path, label_names)
            # Auto-save alongside the CSV
            auto_save = str(Path(landmark_csv_path).with_suffix(".pkl"))
            lm_classifier.save(auto_save)
            LOGGER.info(f"[MediaPipe] Classifier auto-saved to {auto_save}")
        else:
            LOGGER.warning(
                "[MediaPipe] No classifier found. Landmark overlay will be "
                "shown but YOLOv5 labels will be used for classification.\n"
                "  → To train a classifier, collect a landmarks CSV with "
                "--save-landmarks and re-run with --landmark-csv-path."
            )
            lm_classifier = LandmarkClassifier(label_names)  # empty, won't predict

        # Prepare landmark CSV writer for data collection
        if save_landmarks:
            lm_csv_path_out = save_dir / "landmarks_collected.csv"
            lm_csv_file     = open(lm_csv_path_out, "w", newline="")
            lm_csv_writer   = csv.writer(lm_csv_file)
            lm_csv_writer.writerow(
                ["label"] + [f"{ax}{i}" for i in range(21) for ax in ("x","y","z")]
            )
            LOGGER.info(f"[MediaPipe] Saving landmarks to {lm_csv_path_out}")

    # ── Dataloader ───────────────────────────────────────────────────────
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset  = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt,
                               vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,
                             vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ── Warmup ───────────────────────────────────────────────────────────
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (
        Profile(device=device), Profile(device=device), Profile(device=device)
    )

    csv_path = save_dir / "predictions.csv"

    def write_to_csv(image_name, prediction, confidence):
        data_row = {"Image Name": image_name, "Prediction": prediction,
                    "Confidence": confidence}
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)

    # ── Inference loop ───────────────────────────────────────────────────
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize else False
            )
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    p_ = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    pred = p_ if pred is None else torch.cat((pred, p_), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p          = Path(p)
            save_path  = str(save_dir / p.name)
            txt_path   = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )
            s         += "{:g}x{:g} ".format(*im.shape[2:])
            gn         = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc        = im0.copy() if save_crop else im0
            annotator  = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n  = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c_idx       = int(cls)
                    yolo_label  = names[c_idx]
                    confidence  = float(conf)
                    conf_str    = f"{confidence:.2f}"

                    # ── MediaPipe landmark stage ──────────────────────────
                    final_label = yolo_label  # default: use YOLO prediction
                    final_conf  = confidence

                    if use_mediapipe and hand_detector is not None:
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        fh, fw = im0.shape[:2]

                        # Add a small padding around the detected hand box
                        pad   = 20
                        x1p   = max(0,  x1 - pad)
                        y1p   = max(0,  y1 - pad)
                        x2p   = min(fw, x2 + pad)
                        y2p   = min(fh, y2 + pad)
                        crop  = im0[y1p:y2p, x1p:x2p]

                        if crop.size > 0:
                            lm_vec = extract_landmarks(hand_detector, crop)

                            if lm_vec is not None:
                                # ── Classify with MLP if available ───────
                                if lm_classifier.is_ready():
                                    lm_idx, lm_conf = lm_classifier.predict(lm_vec)
                                    if lm_idx is not None and lm_conf > 0.4:
                                        label_list = (
                                            [names[k] for k in sorted(names.keys())]
                                            if isinstance(names, dict) else list(names)
                                        )
                                        if lm_idx < len(label_list):
                                            final_label = label_list[lm_idx]
                                            final_conf  = lm_conf

                                # ── Draw MediaPipe skeleton ───────────────
                                if show_skeleton:
                                    im0 = draw_landmarks_on_frame(
                                        mp_module, im0, hand_detector, xyxy
                                    )

                                # ── Save landmarks for future training ────
                                if save_landmarks and lm_csv_writer is not None:
                                    row = [c_idx] + lm_vec.tolist()
                                    lm_csv_writer.writerow(row)

                                # ── Overlay confidence comparison ─────────
                                cmp_txt = (
                                    f"YOLO:{yolo_label}({confidence:.2f}) | "
                                    f"MP:{final_label}({final_conf:.2f})"
                                )
                                cv2.putText(
                                    im0, cmp_txt,
                                    (max(0, int(xyxy[0])), max(15, int(xyxy[1]) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 255, 255), 1, cv2.LINE_AA,
                                )
                    # ─────────────────────────────────────────────────────

                    if save_csv:
                        write_to_csv(p.name, final_label, f"{final_conf:.2f}")

                    if save_txt:
                        coords = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1).tolist()
                        ) if save_format == 0 else (
                            torch.tensor(xyxy).view(1, 4) / gn
                        ).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:
                        display_label = (
                            None if hide_labels
                            else (
                                final_label if hide_conf
                                else f"{final_label} {final_conf:.2f}"
                            )
                        )
                        annotator.box_label(xyxy, display_label, color=colors(c_idx, True))

                    if save_crop:
                        save_one_box(
                            xyxy, imc,
                            file=save_dir / "crops" / names[c_idx] / f"{p.stem}.jpg",
                            BGR=True,
                        )

            im0 = annotator.result()

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if vid_cap else im0.shape[1]
                        h   = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}"
            f"{dt[1].dt * 1e3:.1f}ms"
        )

    # ── Cleanup ──────────────────────────────────────────────────────────
    if lm_csv_file:
        lm_csv_file.close()
        LOGGER.info(f"[MediaPipe] Landmark CSV saved to {save_dir / 'landmarks_collected.csv'}")

    if use_mediapipe and hand_detector:
        hand_detector.close()

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image "
        f"at shape {(1, 3, *imgsz)}" % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to "
            f"{save_dir / 'labels'}"
        ) if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_opt():
    parser = argparse.ArgumentParser()

    # ── Original YOLOv5 args (unchanged) ────────────────────────────────
    parser.add_argument("--weights", nargs="+", type=str,
                        default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str,
                        default=ROOT / "data/images")
    parser.add_argument("--data", type=str,
                        default="C:/Users/suhai/Desktop/YOLO/data.yaml")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+",
                        type=int, default=[640])
    parser.add_argument("--conf-thres",   type=float, default=0.25)
    parser.add_argument("--iou-thres",    type=float, default=0.45)
    parser.add_argument("--max-det",      type=int,   default=1000)
    parser.add_argument("--device",       default="")
    parser.add_argument("--view-img",     action="store_true")
    parser.add_argument("--save-txt",     action="store_true")
    parser.add_argument("--save-format",  type=int, default=0)
    parser.add_argument("--save-csv",     action="store_true")
    parser.add_argument("--save-conf",    action="store_true")
    parser.add_argument("--save-crop",    action="store_true")
    parser.add_argument("--nosave",       action="store_true")
    parser.add_argument("--classes",      nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment",      action="store_true")
    parser.add_argument("--visualize",    action="store_true")
    parser.add_argument("--update",       action="store_true")
    parser.add_argument("--project",      default=ROOT / "runs/detect")
    parser.add_argument("--name",         default="exp")
    parser.add_argument("--exist-ok",     action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels",  default=False, action="store_true")
    parser.add_argument("--hide-conf",    default=False, action="store_true")
    parser.add_argument("--half",         action="store_true")
    parser.add_argument("--dnn",          action="store_true")
    parser.add_argument("--vid-stride",   type=int, default=1)

    # ── NEW MediaPipe args ───────────────────────────────────────────────
    parser.add_argument(
        "--use-mediapipe", action="store_true",
        help="Enable MediaPipe hand landmark pipeline on top of YOLOv5",
    )
    parser.add_argument(
        "--landmark-clf-path", type=str, default="",
        help="Path to a pre-trained LandmarkClassifier .pkl file",
    )
    parser.add_argument(
        "--landmark-csv-path", type=str, default="",
        help="Path to a landmarks CSV to auto-train the classifier",
    )
    parser.add_argument(
        "--save-landmarks", action="store_true",
        help="Save extracted landmark vectors to CSV (for building training data)",
    )
    parser.add_argument(
        "--mp-model-path", type=str, default="",
        help="Path to hand_landmarker.task model (auto-downloaded if not set)",
    )
    parser.add_argument(
        "--mp-conf", type=float, default=0.5,
        help="MediaPipe hand detection confidence threshold",
    )
    parser.add_argument(
        "--no-skeleton", dest="show_skeleton", action="store_false",
        help="Disable drawing the MediaPipe skeleton overlay",
    )
    parser.set_defaults(show_skeleton=True)

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)