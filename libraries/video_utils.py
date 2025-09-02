import os
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# ===============================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ===============================
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def enhance_contrast_brightness(frame,level):

    frame = apply_clahe(frame)

    if level == 1:
        alpha, beta = 1.1, 10   # Poco contrasto, poca luce
    elif level == 2:
        alpha, beta = 1.5, 30   # Medio contrasto/luminosità
    elif level == 3:
        alpha, beta = 5.0, 50   # Più aggressivo
    else:
        raise ValueError("level must be 1, 2, or 3")

    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# ===============================
# Utility functions
# ===============================
def get_video_properties(cap: cv2.VideoCapture):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_width, frame_height, fps


def draw_bounding_boxes(frame, results, builder, keypoint_colors, show_all_boxes=True, show_confidences=True):
    """
    Disegna i bounding box e le etichette sul frame.
    """
    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = builder.classes_fd[int(cls_id.item())].lower()
        if not show_all_boxes and label != "head":
            continue

        color = keypoint_colors.get(label, (255, 255, 255))  # fallback bianco
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {conf.item() * 100:.1f}%" if show_confidences else label
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
    return frame

# ===============================
# Video processing (generalized)
# ===============================
def process_video(input_video_path: str,
                  builder,
                  clf,
                  use_filters: bool = False,
                  show_confidences: bool = True,
                  show_all_boxes: bool = True,
                  default_fps: int = 20,
                  verbose: bool = False):
    """
    Processa un video con o senza filtri (in base a use_filters).
    """
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error in video opening")
        return
    else:
        print("Video uploaded correctly")

    base, _ = os.path.splitext(input_video_path)
    suffix = "_pred_with_filters.mp4" if use_filters else "_pred_without_filters.mp4"
    output_video_path = base + suffix

    frame_width, frame_height, fps = get_video_properties(cap)
    if fps > default_fps:
        fps = default_fps
    print(f"FPS processing: {fps}")

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    keypoint_colors = {
        "eye": (255, 165, 0),       # orange
        "nose": (127, 255, 212),    # aqua green
        "mouth": (255, 0, 255),     # fuchsia
        "head": (0, 0, 255)         # red
    }

    # --- Variabili condivise ---
    required_kpts = {"eye", "head", "nose", "mouth", "eye1", "eye2"}
    history = deque(maxlen=25)
    stable_success_counter = 0
    min_stable_success_frames = 100
    level = 0  # per i filtri
    boxes_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Filtri (solo se richiesto) ---
        if use_filters and level > 0:
            frame = enhance_contrast_brightness(frame, level)
            if verbose:
                print(f"🔧 Applying filter level {level}")

        # --- Prediction YOLO ---
        results = builder.model_fd(frame, conf=0.3, verbose=False)[0]
        kpt = builder.features_extractor(results.boxes)

        # Conta box validi
        count_valid_boxes = sum(1 for k in required_kpts if k in kpt and kpt[k] != (-1, -1))
        boxes_per_frame.append(count_valid_boxes)

        # --- Success rate (solo se use_filters=True) ---
        if use_filters:
            has_any_keypoint = any(k in kpt and kpt[k] != (-1, -1) and k!="head" for k in required_kpts)
            history.append(1 if has_any_keypoint else 0)
            success_rate = sum(history) / len(history) if history else 0

            if has_any_keypoint:
                stable_success_counter += 1
            else:
                stable_success_counter = 0

            if success_rate > 0.7 and level > 0:
                if stable_success_counter >= min_stable_success_frames:
                    if verbose:
                        print(f"Buoni box frequenti ({success_rate*100:.1f}%) → disattivo filtro")
                    level = max(0, level - 1)
                    stable_success_counter = 0
            elif success_rate < 0.3 and level < 3:
                if verbose:
                    print(f"Success rate basso ({success_rate*100:.1f}%) → aumento filtro a livello {level}")
                level = min(3, level + 1)
                stable_success_counter = 0

        # --- Disegno bounding box ---
        draw_bounding_boxes(frame, results, builder, keypoint_colors, show_all_boxes, show_confidences)

        # --- Feature extraction & prediction ---
        train = builder.create_embedding_for_video(
            kpt, flags=True, positions=True, geometric_info=True, positions_normalized=True
        ).reshape(1, -1)

        pred = clf.predict(train)[0]
        if hasattr(clf, "predict_proba"):
            class_index = np.where(clf.classes_ == pred)[0][0]
            prob = clf.predict_proba(train)[0][class_index]
        else:
            prob = 1.0

        if use_filters:
            label_text = f"{'Safe' if pred == 1 else 'In Danger'} ({prob * 100:.1f}%)"
        else:
            label_text = f"{'Safe' if pred == 1 else 'In Danger'} ({prob * 100:.1f}%)"

        color = (0, 255, 0) if (("Safe" in label_text) or ("Safe" in label_text)) else (0, 0, 255)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Video Prediction", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return boxes_per_frame

# ===============================
# Stats & Visualization Helpers
# ===============================
def count_zero_nonzero(lst):
    zeros = sum(1 for x in lst if x == 0)
    nonzeros = len(lst) - zeros
    return [zeros, nonzeros]


def plot_comparison(valid_boxes_per_frame_with_filter, valid_boxes_per_frame_without_filter):
    """Pie and bar comparison between filtered and unfiltered runs."""
    counts_with = count_zero_nonzero(valid_boxes_per_frame_with_filter)
    counts_without = count_zero_nonzero(valid_boxes_per_frame_without_filter)

    labels = ['0 boxes', '≥1 boxes']
    colors = ['lightgray', 'lightgreen']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(counts_with, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axs[0].set_title('With Filter')

    axs[1].pie(counts_without, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axs[1].set_title('Without Filter')

    plt.suptitle("Bounding Box Detection per Frame")
    plt.show()

    # Bar chart
    length = min(len(valid_boxes_per_frame_with_filter), len(valid_boxes_per_frame_without_filter))
    data_with = valid_boxes_per_frame_with_filter[:length]
    data_without = valid_boxes_per_frame_without_filter[:length]

    x = np.arange(length)
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, data_with, width, label='With Filter', alpha=0.7)
    plt.bar(x + width/2, data_without, width, label='Without Filter', alpha=0.7)
    plt.xlabel('Frame number')
    plt.ylabel('Number of detected keypoints')
    plt.title('Bounding boxes per frame: With vs Without Filter')
    plt.legend()
    plt.show()

    # Totals
    total_with_filter = sum(valid_boxes_per_frame_with_filter)
    total_without_filter = sum(valid_boxes_per_frame_without_filter)

    print(f"Total with filter: {total_with_filter}")
    print(f"Total without filter: {total_without_filter}")

    labels = ['With Filter', 'Without Filter']
    totals = [total_with_filter, total_without_filter]

    plt.bar(labels, totals, color=['skyblue', 'salmon'])
    plt.title("Total Bounding Boxes Detected")
    plt.ylabel("Number of Bounding Boxes")
    plt.show()