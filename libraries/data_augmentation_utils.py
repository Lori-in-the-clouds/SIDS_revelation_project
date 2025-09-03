import pathlib
from pathlib import Path
import re
import cv2
import albumentations as A
import numpy as np

class DataAugmentation:
    def __init__(self, filters: list[A.Compose], dataset_path: pathlib.Path, kpt: bool = False, bbox: bool = False):
        self.filters = []
        self.dataset_path = Path(dataset_path)
        self.there_are_bbox = bbox
        self.there_are_kpt = kpt

        self.bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels_bbox']) if bbox else None
        self.keypoint_params = A.KeypointParams(format='xy', label_fields=['keypoint_labels'],
                                                remove_invisible=False) if kpt else None

        for f in filters:
            self.add_filter(f)

    def add_filter(self, filter):
        transforms = filter.transforms if isinstance(filter, A.Compose) else [filter]
        filter = A.Compose(
            transforms,
            bbox_params=self.bbox_params,
            keypoint_params=self.keypoint_params
        )
        self.filters.append(filter)

    @staticmethod
    def clip_yolo_bbox_coords(bbox):
        cx, cy, w, h = bbox
        # Converte in formato (x_min, y_min, x_max, y_max)
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

        # Effettua il clipping
        x_min = np.clip(x_min, 0.0, 1.0)
        y_min = np.clip(y_min, 0.0, 1.0)
        x_max = np.clip(x_max, 0.0, 1.0)
        y_max = np.clip(y_max, 0.0, 1.0)

        # Riconverte in formato YOLO (cx, cy, w, h)
        new_cx = (x_min + x_max) / 2
        new_cy = (y_min + y_max) / 2
        new_w = x_max - x_min
        new_h = y_max - y_min

        return [new_cx, new_cy, new_w, new_h]

    def apply_transformation(self, image_input, image_output, mode, label_input, label_output):
        image = cv2.imread(str(image_input))
        if image is None:
            print(f"Image not found or not valid: {image_input}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        bboxes = []
        class_labels_bbox = []
        keypoints_yolo = []
        keypoint_labels = []
        keypoints_visibility = []

        if Path(label_input).exists():
            with open(label_input, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        class_id = int(parts[0])

                        if self.there_are_bbox and len(parts) >= 5:
                            bbox = list(map(float, parts[1:5]))
                            clipped_bbox = self.clip_yolo_bbox_coords(bbox)
                            bboxes.append(clipped_bbox)
                            class_labels_bbox.append(class_id)

                        if self.there_are_kpt and len(parts) > 5:
                            kpts_and_vis = list(map(float, parts[5:]))
                            num_kpts_per_object_in_file = (len(kpts_and_vis) // 3)

                            for i in range(num_kpts_per_object_in_file):
                                x_norm = kpts_and_vis[i * 3]
                                y_norm = kpts_and_vis[i * 3 + 1]
                                visibility = kpts_and_vis[i * 3 + 2]

                                keypoints_yolo.append([x_norm * width, y_norm * height])
                                keypoints_visibility.append(visibility)
                                keypoint_labels.append(class_id)
                    except (ValueError, IndexError) as e:
                        print(f"Riga mal formata in {label_input}: {line.strip()} | Errore: {e}")
                        continue

        transform_kwargs = {'image': image}
        if self.there_are_bbox:
            transform_kwargs['bboxes'] = bboxes
            transform_kwargs['class_labels_bbox'] = class_labels_bbox
        if self.there_are_kpt:
            transform_kwargs['keypoints'] = keypoints_yolo
            transform_kwargs['keypoint_labels'] = keypoint_labels

        try:
            augmented = self.filters[mode](**transform_kwargs)
        except Exception as e:
            print(f"Error in Albumentation in {Path(image_input).name}: {e}")
            return

        aug_height, aug_width = augmented['image'].shape[:2]

        augmented_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_output, augmented_bgr)

        # === LOGICA DI SCRITTURA CORRETTA E PIÙ ROBUSTA ===
        with open(label_output, 'w', encoding='utf-8') as f:
            augmented_bboxes = augmented.get('bboxes', [])

            # Non scrivere nulla se non ci sono bounding box da scrivere
            if not self.there_are_bbox or not augmented_bboxes:
                return

            augmented_keypoints = augmented.get('keypoints')
            has_augmented_kpts = self.there_are_kpt and augmented_keypoints is not None

            num_objects = len(augmented_bboxes)
            num_kpts_per_object = 0
            if has_augmented_kpts:
                # Calcola il numero di keypoint per oggetto in base ai keypoint aumentati
                num_kpts_per_object = len(augmented_keypoints) // num_objects if num_objects > 0 else 0

            for i in range(num_objects):
                label = augmented['class_labels_bbox'][i]
                bbox = augmented_bboxes[i]
                bbox_str = " ".join(f"{v:.6f}" for v in bbox)

                final_line_parts = [str(label), bbox_str]

                if has_augmented_kpts and num_kpts_per_object > 0:
                    start_index = i * num_kpts_per_object
                    end_index = start_index + num_kpts_per_object

                    # Controlla se gli indici sono validi
                    if end_index > len(augmented_keypoints):
                        # Questo caso non dovrebbe verificarsi se la logica è corretta,
                        # ma è un ulteriore livello di protezione.
                        continue

                    object_kpts_transformed = augmented_keypoints[start_index:end_index]

                    # Usa la lista di visibilità originale, ma con un controllo di sicurezza
                    # per evitare errori se le liste non si allineano perfettamente
                    if len(keypoints_visibility) > start_index and len(keypoints_visibility) >= end_index:
                        object_vis_original = keypoints_visibility[start_index:end_index]
                    else:
                        object_vis_original = [0] * num_kpts_per_object

                    for idx, kp in enumerate(object_kpts_transformed):
                        x_pix, y_pix = kp
                        x_norm = min(max(x_pix / aug_width, 0), 1)
                        y_norm = min(max(y_pix / aug_height, 0), 1)
                        v = 1 if 0 <= x_norm <= 1 and 0 <= y_norm <= 1 and object_vis_original[idx] > 0 else 0
                        final_line_parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", str(int(v))])

                f.write(" ".join(final_line_parts) + '\n')

    def add_items_in_dataset(self, dir_path, mode):
        dir_images = Path(dir_path) / "images"
        dir_labels = Path(dir_path) / "labels"

        if not dir_images.is_dir() or not dir_labels.is_dir():
            print(f"Directory doesnt' found: {dir_path}")
            return

        images = sorted(f for f in dir_images.iterdir() if f.is_file() and not re.search(r"_t\d+\.jpg$", f.name))
        for img in images:
            label_path = dir_labels / f"{img.stem}.txt"
            if not label_path.exists():
                print(f"Missing label for {img.name}")
                continue
            self.apply_transformation(
                image_input=img,
                image_output=str(dir_images / f"{img.stem}_t{mode}.jpg"),
                label_input=label_path,
                label_output=str(dir_labels / f"{img.stem}_t{mode}.txt"),
                mode=mode
            )

    def clean_dataset(self):
        for subdir in ['test', 'train', 'valid']:
            dir_images = self.dataset_path / subdir / "images"
            dir_labels = self.dataset_path / subdir / "labels"
            if dir_images.is_dir() and dir_labels.is_dir():
                for f in list(dir_images.iterdir()) + list(dir_labels.iterdir()):
                    if re.search(r"_t\d+\.(jpg|txt)$", f.name):
                        f.unlink(missing_ok=True)
        print("✅ Dataset cleaning completed.")


    def apply_data_augmentation(self):
        self.clean_dataset()
        for i in range(len(self.filters)):
            print(f"➡️ The enhancement filter was applied {i + 1}/{len(self.filters)}...")
            for subdir in ['test', 'train', 'valid']:
                dir_path = self.dataset_path / subdir
                self.add_items_in_dataset(str(dir_path), i)
            print(f"✅ Filter {i + 1} applied at all subdirectories")
        print("Data augmentation completed!")
