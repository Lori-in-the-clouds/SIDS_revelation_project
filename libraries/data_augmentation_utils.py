import pathlib
import albumentations as A
from pathlib import Path
import re

class DataAugmentation:
    def __init__(self,filters:list[A.Compose],dataset_path:pathlib.Path):
        self.filters = []
        self.filters += filters
        self.dataset_path = dataset_path

    def add_filter(self, filter:A.Compose):
        self.filters.append(filter)

    def normalize_yolo_bbox(self,bbox, epsilon=1e-7):
        cx, cy, w, h = bbox

        cx = round(cx, 7)
        cy = round(cy, 7)
        w = round(w, 7)
        h = round(h, 7)

        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        x_max = min(1.0 - epsilon, x_max)
        y_max = min(1.0 - epsilon, y_max)

        new_cx = (x_min + x_max) / 2
        new_cy = (y_min + y_max) / 2
        new_w = x_max - x_min
        new_h = y_max - y_min

        new_w = max(0.0, new_w)
        new_h = max(0.0, new_h)

        return [new_cx, new_cy, new_w, new_h]

    def add_items_in_dataset(self,dir_path, mode):
        dir_images = Path(dir_path) / "images"
        dir_labels = Path(dir_path) / "labels"

        images = sorted(
            f for f in dir_images.iterdir()
            if f.is_file() and not re.search(r"_t\d+\.jpg$", f.name)
        )

        for image in images:
            image_stem = image.stem
            label_path = dir_labels / f"{image_stem}.txt"

            if not label_path.exists():
                print(f"⚠️ Label mancante per {image.name}, salto.")
                continue

            image_output = dir_images / f"{image_stem}_t{mode}.jpg"
            label_output = dir_labels / f"{image_stem}_t{mode}.txt"

            try:
                self.apply_transformation(
                    image_input=str(image),
                    image_output=str(image_output),
                    label_input=str(label_path),
                    label_output=str(label_output),
                    mode=mode
                )
            except Exception as e:
                print(f"❌ Errore su {image.name}: {e}")


    def apply_transformation(self,image_input, image_output, mode, label_input, label_output):
        import cv2

        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(label_input, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))

                    normalized_bbox = self.normalize_yolo_bbox(bbox)
                    bboxes.append(normalized_bbox)
                    class_labels.append(class_id)

        augmented = self.filters[mode](
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        assert len(augmented['bboxes']) == len(augmented['class_labels']), \
            f"bbox-label length mismatch: {len(augmented['bboxes'])} vs {len(augmented['class_labels'])}"

        augmented_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_output, augmented_bgr)

        clipped_bboxes = []
        clipped_labels = []
        for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
            bbox = [min(max(x, 0.0), 1.0) for x in bbox]
            x, y, w, h = bbox
            if w > 0.001 and h > 0.001:
                clipped_bboxes.append([x, y, w, h])
                clipped_labels.append(label)

        with open(label_output, 'w', encoding='utf-8') as f:
            for label, bbox in zip(clipped_labels, clipped_bboxes):
                bbox_str = " ".join(f"{v:.6f}" for v in bbox)
                f.write(f"{label} {bbox_str}\n")

    def clean_dataset(self):

        for dir in ["/test","/train","/valid"]:

            dir_images = Path(self.dataset_path+dir + "/images")
            dir_labels = Path(self.dataset_path+dir + "/labels")

            images = sorted(f for f in dir_images.iterdir() if f.is_file())
            labels = sorted(f for f in dir_labels.iterdir() if f.is_file())

            for image in images:
                if re.search(r"_t\d+\.jpg$", image.name):
                    image.unlink(missing_ok=True)

            for label in labels:
                if label:
                    if re.search(r"_t\d+\.txt$", label.name):
                        label.unlink(missing_ok=True)

    def apply_data_augmentation(self):

        self.clean_dataset()

        for i in range(0, len(self.filters)):
            self.add_items_in_dataset(str(self.dataset_path + "/test"), i)
            self.add_items_in_dataset(str(self.dataset_path + "/train"), i)
            self.add_items_in_dataset(str(self.dataset_path + "/valid"), i)








