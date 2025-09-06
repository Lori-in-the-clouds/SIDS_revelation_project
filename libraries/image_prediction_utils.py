from libraries.embeddings_utils import *
from libraries.video_utils import *

def image_prediction(image_path, builder: EmbeddingBuilder, clf, show_all_boxes=True, show_confidences=True,thickness_line=1, thickness_point=2,show_bounding_box=True,show_keypoints=True,show_prediction=True):

    # Upload Image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Immagine non trovata: {image_path}")

    # --- Prediction YOLO ---
    results_fd = builder.model_fd(frame, conf=0.7, verbose=False)[0]
    results_pe = builder.model_pe(frame, conf=0.7, verbose=False)[0]

    # --- Keypoints ---
    kpt_fd = builder.features_extractor(results_fd.boxes)
    kpt_pe = builder.features_extractor_keypoints(results_pe)
    kpt = {**kpt_fd, **kpt_pe}

    # List keypoints
    expected_kpts = [
        "eye1", "eye2", "nose", "mouth", "head",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]

    for k in expected_kpts:
        if k not in kpt:
            kpt[k] = (-1, -1)

    # --- Draw bounding box ---
    keypoint_colors = {"head": (0, 50, 200), "eye": (255,165,0), "nose": (127,255,212)}
    if show_bounding_box:
        frame = draw_bounding_boxes(frame, results_fd, builder, keypoint_colors, show_all_boxes, show_confidences=show_confidences)

    # --- Draw keypoints ---
    keypoints_order = [
        "nose_k", "left_eye_k", "right_eye_k", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    keypoints_list = [kpt.get(k, None) if kpt.get(k, (-1, -1)) != (-1, -1) else None for k in keypoints_order]
    if show_keypoints:
        frame = draw_keypoints_on_frame(frame, keypoints_list, number=False,thickness_line=thickness_line, thickness_point=thickness_point)

    # --- Prediction ---
    train = builder.create_embedding_for_video(
        kpt, flags=True, positions=True, geometric_info=True, positions_normalized=True,
        k_positions_normalized=True, k_geometric_info=True,
    ).reshape(1, -1)

    pred = clf.predict(train)[0]
    if hasattr(clf, "predict_proba"):
        class_index = np.where(clf.classes_ == pred)[0][0]
        prob = clf.predict_proba(train)[0][class_index]
    else:
        prob = 1.0

    label_text = f"{'Safe' if pred == 0 else 'In Danger'} ({prob * 100:.1f}%)"
    color = (0, 255, 0) if "Safe" in label_text else (0, 0, 255)

    if show_prediction:
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Save image ---
    base, ext = os.path.splitext(image_path)
    image_output_path = f"{base}_pred{ext}"
    success = cv2.imwrite(image_output_path, frame)
    if not success:
        raise IOError(f"Errore nel salvataggio di {image_output_path}")

    return