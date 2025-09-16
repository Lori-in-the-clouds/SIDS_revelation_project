import os
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch


def draw_keypoints_on_frame(img, keypoints_list, number: bool = False, thickness_line=2, thickness_point=3):
    h, w = img.shape[:2]

    skeleton = [(0, 6),(0, 5),(6,8),(0,1),(0,2),(6,5),(6,4),(5,3),(4,2),(3,1),
                (6,12),(5,7),(5,11),(7,9),(8,10),(12,11),(12,14),(14,16),(11,13),(13,15)]


    # Disegna punti
    for idx, kp in enumerate(keypoints_list):
        if kp is not None:
            x_px = int(kp[0] * w)
            y_px = int(kp[1] * h)
            cv2.circle(img, (x_px, y_px), thickness_point, (0, 0, 255), -1)
            if number:
                cv2.putText(img, str(idx), (x_px + 3, y_px - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Disegna linee scheletro
    for start, end in skeleton:
        if keypoints_list[start] is not None and keypoints_list[end] is not None:
            x1, y1 = int(keypoints_list[start][0] * w), int(keypoints_list[start][1] * h)
            x2, y2 = int(keypoints_list[end][0] * w), int(keypoints_list[end][1] * h)

            # colori scheletro
            if (start == 6 and end == 5) or (start == 6 and end == 12) or (start == 5 and end == 11) or (start == 12 and end == 11):
                color = (255, 0, 255)
            elif (start == 12 and end == 14) or (start == 11 and end == 13) or (start == 14 and end == 16) or (start == 13 and end == 15):
                color = (0, 165, 255)
            elif (start == 6 and end == 8) or (start == 8 and end == 10) or (start == 5 and end == 7) or (start == 7 and end == 9):
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)

            cv2.line(img, (x1, y1), (x2, y2), color, thickness_line)

    return img


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
                  show_all_kpt: bool = True,
                  default_fps: int = 20,
                  verbose: bool = False,
                  upper_thresh=0.65, # decrease filter if above
                  lower_thresh=0.35): # increase filter if below

    def count_valid(kpts_set):
        """Count valid keypoints in a given set."""
        return sum(1 for k in kpts_set if k in kpt and kpt[k] != (-1, -1))

    #Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {input_video_path}")

    base, _ = os.path.splitext(input_video_path)
    suffix = "_pred_with_filters.mp4" if use_filters else "_pred_without_filters.mp4"
    output_video_path = base + suffix

    # Get video properties
    frame_width, frame_height, fps = get_video_properties(cap)
    fps = min(fps, default_fps)
    print(f"FPS processing: {fps}")

    # Video writer
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Keypoint colors for drawing
    keypoint_colors = {
        "eye": (255, 165, 0),       # orange
        "nose": (127, 255, 212),    # aqua green
        "mouth": (255, 0, 255),     # fuchsia
        "head": (0, 50, 200),     # blue
        "left_shoulder": (0, 0, 255), # red
        "right_shoulder": (0, 0, 255),
        "left_hip": (0, 0, 255),
        "right_hip": (0, 0, 255),
        "left_knee": (0, 0, 255),
        "right_knee": (0, 0, 255),
        "left_ankle": (0, 0, 255),
        "right_ankle": (0, 0, 255)
    }

    # Keypoints required for counting
    required_kpts = {  "eye", "head", "nose", "mouth",
                        "eye1", "eye2","left_shoulder", "right_shoulder",
                        "left_hip", "right_hip",
                        "left_knee", "right_knee",
                        "left_ankle", "right_ankle"}
    history = deque(maxlen=25)
    level = 0  # per i filtri
    boxes_per_frame = []
    keypoints_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Apply adaptive filter if enabled ---
        if use_filters and level > 0:
            frame = enhance_contrast_brightness(frame, level)
            if verbose:
                print(f"🔧 Applying filter level {level}")

        # --- Prediction YOLO ---
        results1 = builder.model_fd(frame, conf=0.3, verbose=False)[0]
        results2 = builder.model_pe(frame, conf=0.3, verbose=False)[0]

        #Extract keypoints from face detection and pose estimation
        kpt_fe = builder.features_extractor(results1.boxes)
        kpt_pe = builder.features_extractor_keypoints(results2)
        kpt = {**kpt_fe, **kpt_pe}

        expected_kpts = [
            "eye1", "eye2", "nose", "mouth", "head","left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow","left_wrist", "right_wrist","left_hip", "right_hip",
            "left_knee", "right_knee","left_ankle", "right_ankle"]

        for k in expected_kpts:
            if k not in kpt:
                kpt[k] = (-1, -1)

        # --- Count valid bounding boxes and keypoints ---
        required_boxes = ["eye", "head", "nose", "mouth"]

        count_valid_boxes = sum(1 for k in required_boxes if k in kpt and kpt[k] != (-1, -1))
        boxes_per_frame.append(count_valid_boxes)

        required_kpts = ["left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow","left_wrist", "right_wrist","left_hip", "right_hip",
            "left_knee", "right_knee","left_ankle", "right_ankle","left_ear", "right_ear"]

        num_valid_keypoints = sum(1 for k in required_kpts if kpt[k] != (-1, -1))
        keypoints_per_frame.append(num_valid_keypoints)

        # --- Adaptive filter score ---
        if use_filters:
            # Compute brightness and contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_norm = np.mean(gray) / 255
            contrast_norm = np.std(gray) / 127
            visual_score = (brightness_norm + contrast_norm) / 2

            # Keypoint score with more weight to critical points
            critical = {"head"}
            weight_critical = 4
            secondary = {"left_hip", "right_hip"}
            weight_secondary = 0.5
            others = {"left_hip", "right_hip","left_knee", "right_knee", "left_ankle", "right_ankle"}
            weight_others = 0.3

            keypoint_score = (weight_critical * count_valid(critical) +
                             weight_secondary * count_valid(secondary) +
                            weight_others * count_valid(others)) / (weight_critical * len(critical) + weight_secondary * len(secondary) + weight_others * len(others))

            # Final frame score: weighted combination
            frame_score = 0.75 * keypoint_score + 0.25 * visual_score

            # Update history and compute weighted averagey
            history.append(frame_score)
            success_rate = np.average(history, weights=np.linspace(0.1, 1.0, len(history)))

            # --- Hysteresis thresholds ---
            if success_rate > upper_thresh and level > 0:
                    if verbose:
                        print(f"Buoni box frequenti ({success_rate*100:.1f}%) → disattivo filtro")
                    level = max(0, level - 1)
            elif success_rate < lower_thresh and level < 3:
                if verbose:
                    print(f"Success rate basso ({success_rate*100:.1f}%) → aumento filtro a livello {level}")
                level = min(3, level + 1)

        # --- Draw bounding boxes and keypoints ---
        if show_all_boxes:
            draw_bounding_boxes(frame, results1, builder, keypoint_colors, show_all_boxes, show_confidences)

        keypoints_order = [
            "nose_k", "left_eye_k", "right_eye_k", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        keypoints_list = [kpt.get(k, None) if kpt.get(k, (-1, -1)) != (-1, -1) else None for k in keypoints_order]
        if show_all_kpt:
            frame = draw_keypoints_on_frame(frame, keypoints_list, number=False)

        # --- Feature extraction & prediction ---
        train = builder.create_embedding_for_video(
            kpt, flags=True, positions=True, geometric_info=True, positions_normalized=True,
            k_positions_normalized=True, k_geometric_info=True,
        ).reshape(1, -1)

        pred = clf.predict(train)[0]
        prob = clf.predict_proba(train)[0][np.where(clf.classes_ == pred)[0][0]] if hasattr(clf,"predict_proba") else 1.0

        label_text = f"{'Safe' if pred == 0 else 'In Danger'} ({prob * 100:.1f}%)"
        color = (0, 255, 0) if (("Safe" in label_text) or ("Safe" in label_text)) else (0, 0, 255)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output
        cv2.imshow("Video Prediction", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return boxes_per_frame + keypoints_per_frame


def process_video_mlp(input_video_path: str,
                  builder,
                  clf,
                model_mlp,
                  use_filters: bool = False,
                  show_confidences: bool = True,
                  show_all_boxes: bool = True,
                  show_all_kpt: bool = True,
                  default_fps: int = 20,
                  verbose: bool = False,
                  upper_thresh=0.65, # decrease filter if above
                  lower_thresh=0.35): # increase filter if below

    def count_valid(kpts_set):
        """Count valid keypoints in a given set."""
        return sum(1 for k in kpts_set if k in kpt and kpt[k] != (-1, -1))

    #Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {input_video_path}")

    base, _ = os.path.splitext(input_video_path)
    suffix = "_pred_with_filters.mp4" if use_filters else "_pred_without_filters.mp4"
    output_video_path = base + suffix

    # Get video properties
    frame_width, frame_height, fps = get_video_properties(cap)
    fps = min(fps, default_fps)
    print(f"FPS processing: {fps}")

    # Video writer
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Keypoint colors for drawing
    keypoint_colors = {
        "eye": (255, 165, 0),       # orange
        "nose": (127, 255, 212),    # aqua green
        "mouth": (255, 0, 255),     # fuchsia
        "head": (0, 50, 200),     # blue
        "left_shoulder": (0, 0, 255), # red
        "right_shoulder": (0, 0, 255),
        "left_hip": (0, 0, 255),
        "right_hip": (0, 0, 255),
        "left_knee": (0, 0, 255),
        "right_knee": (0, 0, 255),
        "left_ankle": (0, 0, 255),
        "right_ankle": (0, 0, 255)
    }

    # Keypoints required for counting
    required_kpts = {  "eye", "head", "nose", "mouth",
                        "eye1", "eye2","left_shoulder", "right_shoulder",
                        "left_hip", "right_hip",
                        "left_knee", "right_knee",
                        "left_ankle", "right_ankle"}
    history = deque(maxlen=25)
    level = 0  # per i filtri
    boxes_per_frame = []
    keypoints_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Apply adaptive filter if enabled ---
        if use_filters and level > 0:
            frame = enhance_contrast_brightness(frame, level)
            if verbose:
                print(f"🔧 Applying filter level {level}")

        # --- Prediction YOLO ---
        results1 = builder.model_fd(frame, conf=0.3, verbose=False)[0]
        results2 = builder.model_pe(frame, conf=0.3, verbose=False)[0]

        #Extract keypoints from face detection and pose estimation
        kpt_fe = builder.features_extractor(results1.boxes)
        kpt_pe = builder.features_extractor_keypoints(results2)
        kpt = {**kpt_fe, **kpt_pe}

        expected_kpts = [
            "eye1", "eye2", "nose", "mouth", "head","left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow","left_wrist", "right_wrist","left_hip", "right_hip",
            "left_knee", "right_knee","left_ankle", "right_ankle"]

        for k in expected_kpts:
            if k not in kpt:
                kpt[k] = (-1, -1)

        # --- Count valid bounding boxes and keypoints ---
        required_boxes = ["eye", "head", "nose", "mouth"]

        count_valid_boxes = sum(1 for k in required_boxes if k in kpt and kpt[k] != (-1, -1))
        boxes_per_frame.append(count_valid_boxes)

        required_kpts = ["left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow","left_wrist", "right_wrist","left_hip", "right_hip",
            "left_knee", "right_knee","left_ankle", "right_ankle","left_ear", "right_ear"]

        num_valid_keypoints = sum(1 for k in required_kpts if kpt[k] != (-1, -1))
        keypoints_per_frame.append(num_valid_keypoints)

        # --- Adaptive filter score ---
        if use_filters:
            # Compute brightness and contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_norm = np.mean(gray) / 255
            contrast_norm = np.std(gray) / 127
            visual_score = (brightness_norm + contrast_norm) / 2

            # Keypoint score with more weight to critical points
            critical = {"head"}
            weight_critical = 4
            secondary = {"left_hip", "right_hip"}
            weight_secondary = 0.5
            others = {"left_hip", "right_hip","left_knee", "right_knee", "left_ankle", "right_ankle"}
            weight_others = 0.3

            keypoint_score = (weight_critical * count_valid(critical) +
                             weight_secondary * count_valid(secondary) +
                            weight_others * count_valid(others)) / (weight_critical * len(critical) + weight_secondary * len(secondary) + weight_others * len(others))

            # Final frame score: weighted combination
            frame_score = 0.75 * keypoint_score + 0.25 * visual_score

            # Update history and compute weighted averagey
            history.append(frame_score)
            success_rate = np.average(history, weights=np.linspace(0.1, 1.0, len(history)))

            # --- Hysteresis thresholds ---
            if success_rate > upper_thresh and level > 0:
                    if verbose:
                        print(f"Buoni box frequenti ({success_rate*100:.1f}%) → disattivo filtro")
                    level = max(0, level - 1)
            elif success_rate < lower_thresh and level < 3:
                if verbose:
                    print(f"Success rate basso ({success_rate*100:.1f}%) → aumento filtro a livello {level}")
                level = min(3, level + 1)

        # --- Draw bounding boxes and keypoints ---
        if show_all_boxes:
            draw_bounding_boxes(frame, results1, builder, keypoint_colors, show_all_boxes, show_confidences)

        keypoints_order = [
            "nose_k", "left_eye_k", "right_eye_k", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        keypoints_list = [kpt.get(k, None) if kpt.get(k, (-1, -1)) != (-1, -1) else None for k in keypoints_order]
        if show_all_kpt:
            frame = draw_keypoints_on_frame(frame, keypoints_list, number=False)

        # --- Feature extraction & prediction ---
        train = builder.create_embedding_for_video(
            kpt, flags=True, positions=True, geometric_info=True, positions_normalized=True,
            k_positions_normalized=True, k_geometric_info=True,
        ).reshape(1, -1)

        model_mlp.eval()
        train = torch.tensor(train, dtype=torch.float32)

        train = model_mlp(train)

        pred = clf.predict(train.detach().cpu().numpy())[0]
        prob = clf.predict_proba(train.detach().cpu().numpy())[0][np.where(clf.classes_ == pred)[0][0]] if hasattr(clf,"predict_proba") else 1.0

        label_text = f"{'Safe' if pred == 0 else 'In Danger'} ({prob * 100:.1f}%)"
        color = (0, 255, 0) if (("Safe" in label_text) or ("Safe" in label_text)) else (0, 0, 255)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output
        cv2.imshow("Video Prediction", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return boxes_per_frame + keypoints_per_frame

# ===============================
# Stats & Visualization Helpers
# ===============================
def count_zero_nonzero(lst):
    zeros = sum(1 for x in lst if x == 0)
    nonzeros = len(lst) - zeros
    return [zeros, nonzeros]


def plot_comparison(valid_boxes_per_frame_with_filter, valid_boxes_per_frame_without_filter):
    """Pie and bar comparison between filtered and unfiltered runs_keypoints."""
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