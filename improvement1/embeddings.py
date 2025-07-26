import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics import YOLO

from pathlib import Path
import os
import json
import cv2

''' GEOMETRIC FUNCTIONS '''


def compute_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def compute_point_to_line_distance(point, line_start, line_end):
    # Distance from a point to a line defined by line_start and line_end
    return (np.abs(np.cross(np.array(line_end) - np.array(line_start),
                            np.array(line_start) - np.array(point))) / compute_distance(line_start, line_end))


def compute_face_angle(el1, nose, el2):
    # Computes the angle at nose between el1–nose–el2
    vector1 = np.array(el1) - np.array(nose)
    vector2 = np.array(el2) - np.array(nose)
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def normalize(coordinates, head):
    if head == -1:
        return list(coordinates)
    else:
        head_xy = (head[0], head[1])
        return [a / b for a,b in zip(coordinates, head_xy)]


''' FILE MANAGER FUNCTIONS '''


def move_to_project_folder(weights_path: str):
    possible_paths = [
        Path().resolve(),
        Path("/home/terra/Documents/AI_engineering/SIDS-project/python_project/SIDS_revelation_project"),
        Path(" ")  # percorso di Lore
    ]

    for path in possible_paths:
        try:
            full_path = path / weights_path
            if full_path.exists():
                os.chdir(path)
                print(f"Moved to {path}")
                break

        except FileNotFoundError:
            pass
    else:
        raise RuntimeError("No model loaded, all paths were invalid.")


''' EMBEDDING BUILDER CLASS '''


class EmbeddingBuilder:
    def __init__(self, weights_path: str, dataset_path: str):
        move_to_project_folder(weights_path)

        # load YOLOv8 model
        self.model_fd = YOLO(weights_path)
        self.classes_fd = self.model_fd.names

        # save model name
        self.model_n = weights_path[0]

        # select dataset to be processed
        self.dataset = Path(dataset_path)
        self.file_label = {}
        self.dim_dataset = 0
        self.classes_mlp = None

        # prepare keypoints, X, y
        self.keypoints = []
        self.X = []
        self.y = []


    def progress_debug(self, var_to_monitor: list):
        if len(var_to_monitor) % 100 == 0:
            print(
                f"{int(len(var_to_monitor) * 100 / self.dim_dataset)}%-->    {len(var_to_monitor)} / {self.dim_dataset} files processed")

    def extract_dataset_info(self):
        # extract file name- label dict from .json file
        file_imgid = {}
        imgid_label = {}

        file_label = {}
        classes_mlp = {}

        with open(f"{str(self.dataset)}/_annotations.coco.json", "r") as f:
            dataset = json.load(f)

            for img in dataset["images"]:
                file_imgid[img["file_name"]] = img["id"]

            for label in dataset["annotations"]:
                imgid_label[label["image_id"]] = label["category_id"]

            for cls in dataset["categories"]:
                classes_mlp[cls["name"]] = cls["id"]
            del classes_mlp["baby"]
            del classes_mlp["crib"]

        for file, img_id in file_imgid.items():
            if (img_id in imgid_label.keys()) and (imgid_label[img_id] in classes_mlp.values()):
                self.file_label[file] = imgid_label[img_id]
        self.dim_dataset = len(self.file_label)
        self.classes_mlp = classes_mlp

    def keypoints_extractor(self, prediction: Boxes):
        kpt = {}
        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = self.classes_fd[cls.item()]
            class_label = class_label.lower()

            if class_label == "eye":
                if "eye1" in kpt:
                    kpt["eye2"] = (bbox[0], bbox[1])
                else:
                    kpt["eye1"] = (bbox[0], bbox[1])
            elif class_label == "mouth":
                kpt["mouth"] = (bbox[0], bbox[1])
            elif class_label == "nose":
                kpt["nose"] = (bbox[0], bbox[1])
            elif class_label == "head":
                kpt["head"] = (bbox[0], bbox[1], bbox[2], bbox[3])

        return kpt

    def save_keypoints_and_y(self):
        np.save(f"{str(self.dataset)}/improvement1_model{self.model_n}_keypoints.npy", self.keypoints)
        np.save(f"{str(self.dataset)}/improvement1_model{self.model_n}_labels.npy", self.y)

        print(f"keypoints saved in '{str(self.dataset)}/improvement1_model{self.model_n}_keypoints.npy' and labels saved in '{str(self.dataset)}/improvement1_model{self.model_n}_labels.npy")

    def load_keypoints_and_y(self):
        self.keypoints = np.load(f"{str(self.dataset)}/improvement1_model{self.model_n}_keypoints.npy", allow_pickle=True).tolist()
        self.y = np.load(f"{str(self.dataset)}/improvement1_model{self.model_n}_labels.npy", allow_pickle=True).tolist()
        self.dim_dataset = len(self.y)

        print(f"Keypoints and labels loaded succesfully, in particular there are {self.dim_dataset} files in the dataset")

    def process_dataset(self):
        # extract classes_mlp, dim_dataset, file_label dictionary
        self.extract_dataset_info()

        # extract keypoints from each file
        for img in self.dataset.glob("*.jpg"):
            if img.name in self.file_label.keys():
                self.progress_debug(self.y)
                result = self.model_fd(img, conf=0.3, verbose=False)[0]

                kpt = self.keypoints_extractor(result.boxes)
                label = self.file_label[img.name]

                self.keypoints.append(kpt)
                self.y.append(label)

        print(f"FINISHED: {len(self.y)} image processed, keypoints and labels(y) extracted")
        self.save_keypoints_and_y()

    def process_dataset_save_predictions(self):
        # extract classes_mlp, dim_dataset, file_label dictionary
        self.extract_dataset_info()

        output_dir = Path(f"models/{self.model_n}.predictions")
        output_dir.mkdir(exist_ok=True, parents=True)

        # extract keypoints from each file
        for img_path in self.dataset.glob("*.jpg"):
            if img_path.name in self.file_label.keys():
                self.progress_debug(self.y)
                result = self.model_fd(img_path, conf=0.3, verbose=False)[0]

                kpt = self.keypoints_extractor(result.boxes)
                label = self.file_label[img_path.name]

                self.keypoints.append(kpt)
                self.y.append(label)

                # save image with bboxes
                img = cv2.imread(str(img_path))
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # ottieni bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), img)

        print(f"FINISHED: {len(self.y)} image processed, keypoints and labels(y) extracted")
        self.save_keypoints_and_y()

    def embedding_all_features(self):
        X = []
        features = ["flag_eye1","flag_eye2", "flag_nose", "flag_mouth", "x_eye1","y_eye1", "x_eye2", "y_eye2", "x_nose", "y_nose", "x_mouth", "y_mouth", "eye_distance", "face_vertical_length", "face_angle_vertical", "face_angle_horizontal", "symmetry_diff"]

        for kpt, cls in zip(self.keypoints, self.y):
            self.progress_debug(X)
            presence_flags = [
                int('eye1' in kpt),
                int('eye2' in kpt),
                int('nose' in kpt),
                int('mouth' in kpt),
            ]
            eye1 = kpt["eye1"] if presence_flags[0] == 1 else (-1, -1)
            eye2 = kpt["eye2"] if presence_flags[1] == 1 else (-1, -1)
            nose = kpt["nose"] if presence_flags[2] == 1 else (-1, -1)
            mouth = kpt["mouth"] if presence_flags[3] == 1 else (-1, -1)

            # coordinates keypoints
            coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)

            # distance between eyes
            eye_distance = compute_distance(eye1, eye2) if (presence_flags[0] * presence_flags[1]) == 1 else -1

            # vertical face length (nose to mouth)
            face_vertical_length = compute_distance(nose, mouth) if (presence_flags[2] * presence_flags[3]) == 1 else -1

            # angle between eye1 – nose – mouth
            face_angle_vertical = compute_face_angle(eye1, nose, mouth) if (presence_flags[0] * presence_flags[2] * presence_flags[3]) == 1 else -1

            # angle between eye1-nose-eye2
            face_angle_horizontal = compute_face_angle(eye1, nose, eye2) if (presence_flags[0] * presence_flags[2] * presence_flags[1]) == 1 else -1

            # 16: symmetry (difference of eye distances to nose–mouth line)
            symmetry_diff = 0.0
            if (presence_flags[0] * presence_flags[1] * presence_flags[2] * presence_flags[3]) == 1:
                try:
                    eye1_to_axis = compute_point_to_line_distance(eye1, nose, mouth)
                    eye2_to_axis = compute_point_to_line_distance(eye2, nose, mouth)
                    symmetry_diff = abs(eye1_to_axis - eye2_to_axis)
                except:
                    pass

            # final embedding
            embedding = (
                    presence_flags +
                    coordinates +
                    [eye_distance, face_vertical_length, face_angle_vertical, face_angle_horizontal, symmetry_diff]
            )
            X.append(embedding)

        print(f"FINISHED: {len(X)} embedding created")
        return (X, features)


    def embedding_all_features_norm(self):
            X = []
            features = ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth",
                        "x_eye1", "y_eye1", "x_eye2", "y_eye2", "x_nose", "y_nose", "x_mouth", "y_mouth",
                        "x_eye1_norm", "y_eye1_norm", "x_eye2_norm", "y_eye2_norm", "x_nose_norm", "y_nose_norm", "x_mouth_norm", "y_mouth_norm",
                        "eye_distance", "eye_distance_norm", "face_vertical_length", "face_vertical_length_norm",
                        "face_angle_vertical", "face_angle_horizontal", "symmetry_diff", "head_ration"]

            for kpt, cls in zip(self.keypoints, self.y):
                self.progress_debug(X)
                presence_flags = [
                    int('eye1' in kpt),
                    int('eye2' in kpt),
                    int('nose' in kpt),
                    int('mouth' in kpt),
                ]
                head = kpt["head"] if "head" in kpt else -1
                eye1 = kpt["eye1"] if presence_flags[0] == 1 else (-1, -1)
                eye2 = kpt["eye2"] if presence_flags[1] == 1 else (-1, -1)
                nose = kpt["nose"] if presence_flags[2] == 1 else (-1, -1)
                mouth = kpt["mouth"] if presence_flags[3] == 1 else (-1, -1)

                # coordinates keypoints
                coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)

                coordinates_norm = normalize(eye1, head) + normalize(eye2, head) + normalize(nose, head) + normalize(mouth, head)

                # head h/w ration
                head_ration = (head[3] / head[2]) if (head != -1) else -1

                # distance between eyes
                eye_distance = compute_distance(eye1, eye2) if (presence_flags[0] * presence_flags[1]) == 1 else -1
                eye_distance_norm = (eye_distance/head[2]) if (eye_distance!=-1 and head != -1) else -1

                # vertical face length (nose to mouth)
                face_vertical_length = compute_distance(nose, mouth) if (presence_flags[2] * presence_flags[
                    3]) == 1 else -1
                face_vertical_length_norm = (face_vertical_length / head[3]) if (face_vertical_length != -1 and head!=-1) else -1

                # angle between eye1 – nose – mouth
                face_angle_vertical = compute_face_angle(eye1, nose, mouth) if (presence_flags[0] * presence_flags[2] *
                                                                                presence_flags[3]) == 1 else -1

                # angle between eye1-nose-eye2
                face_angle_horizontal = compute_face_angle(eye1, nose, eye2) if (presence_flags[0] * presence_flags[2] *
                                                                                 presence_flags[1]) == 1 else -1

                # 16: symmetry (difference of eye distances to nose–mouth line)
                symmetry_diff = 0.0
                if (presence_flags[0] * presence_flags[1] * presence_flags[2] * presence_flags[3]) == 1:
                    try:
                        eye1_to_axis = compute_point_to_line_distance(eye1, nose, mouth)
                        eye2_to_axis = compute_point_to_line_distance(eye2, nose, mouth)
                        symmetry_diff = abs(eye1_to_axis - eye2_to_axis)
                    except:
                        pass

                # final embedding
                embedding = (
                        presence_flags +
                        coordinates +
                        coordinates_norm +
                        [eye_distance, eye_distance_norm, face_vertical_length, face_vertical_length_norm, face_angle_vertical, face_angle_horizontal, symmetry_diff, head_ration]
                )
                X.append(embedding)

            print(f"FINISHED: {len(X)} embedding created")
            return (X, features)