import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics import YOLO

from pathlib import Path
import os
import json

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
            if class_label == "EYE":
                if "eye1" in kpt:
                    kpt["eye2"] = (bbox[0], bbox[1])
                else:
                    kpt["eye1"] = (bbox[0], bbox[1])
            elif class_label == "MOUTH":
                kpt["mouth"] = (bbox[0], bbox[1])
            elif class_label == "NOSE":
                kpt["nose"] = (bbox[0], bbox[1])
        return kpt

    def save_keypoints_and_y(self):
        np.save(f"{str(self.dataset)}/keypoints.npy", self.keypoints)
        np.save(f"{str(self.dataset)}/labels.npy", self.y)

        print(f"keypoints saved in '{str(self.dataset)}/keypoints.npy' and labels saved in '{str(self.dataset)}/labels.npy")

    def load_keypoints_and_y(self):
        self.keypoints = np.load(f"{str(self.dataset)}/keypoints.npy", allow_pickle=True).tolist()
        self.y = np.load(f"{str(self.dataset)}/labels.npy", allow_pickle=True).tolist()
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

        print(f"FINISHED:\n-> {len(self.y)} image processed, keypoints and labels(y) extracted")
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

        print(f"FINISHED:\n-> {len(X)} embedding created")
        return (X, features)


'''
    # function per calcolare l'angolo tra eye1, nose e eye2 che sono tuple di due float
    def angle_eval(self, eye1, nose, eye2):

        eye1_nose = np.array(eye1) - np.array(nose)
        eye2_nose = np.array(eye2) - np.array(nose)

        cos_theta = np.dot(eye1_nose, eye2_nose) / (np.linalg.norm(eye1_nose) * np.linalg.norm(eye2_nose))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)

        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def embedding1(self, prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (2, 4),
            "MOUTH": 6,
            "NOSE": 8,
            "ANGLE_NOSE_EYES": 10
        }
        vec = np.full(11, -1.0, dtype=float)

        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = classes_fd[cls.item()]

            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]
            elif class_label != "BABY":
                pos = embedding_position[class_label]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]

            pos = embedding_position["ANGLE_NOSE_EYES"]
            eye1 = (vec[embedding_position["EYE"]], vec[embedding_position["EYE"] + 1])
            eye2 = (vec[embedding_position["EYE"] + 2], vec[embedding_position["EYE"] + 3])
            nose = (vec[embedding_position["NOSE"]], vec[embedding_position["NOSE"] + 1])
            if eye1[0] != -1 and eye2[0] != -1 and nose[0] != -1:
                vec[pos] = angle_eval(eye1, nose, eye2)

        return vec

    def embedding2(prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (1, 2),
            "MOUTH": 3,
            "NOSE": 4,
            "ANGLE_NOSE_EYES": 5
        }
        dictionary_eyes_nose = {"EYE": []}

        vec = np.full(6, -1.0, dtype=float)
        for cls in prediction.cls:
            class_label = classes_fd[cls.item()]
            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]
                vec[pos] = 1

                dictionary_eyes_nose["EYE"].append((bbox[0], bbox[1]))

            elif class_label != "BABY":
                pos = embedding_position[class_label]
                vec[pos] = 1

                if class_label == "NOSE":
                    dictionary_eyes_nose["NOSE"] = (bbox[0], bbox[1])

            pos = embedding_position["ANGLE_NOSE_EYES"]

            if len(dictionary_eyes_nose) == 2 and dictionary_eyes_nose["EYE"] == 2:
                eye1 = dictionary_eyes_nose["EYE"][0]
                eye2 = dictionary_eyes_nose["EYE"][1]
                nose = dictionary_eyes_nose["NOSE"]
                vec[pos] = angle_eval(eye1, nose, eye2)
        return vec

    def embedding3(prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (2, 4),
            "MOUTH": 6,
            "NOSE": 8,
            "ANGLE_NOSE_EYES": 10,
            "EYES VISIBILITY": 11
        }
        vec = np.full(12, -1.0, dtype=float)
        vec[embedding_position["EYES VISIBILITY"]] = 0

        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = classes_fd[cls.item()]

            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]
                vec[embedding_position["EYES VISIBILITY"]] += 1
            elif class_label != "BABY":
                pos = embedding_position[class_label]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]

            pos = embedding_position["ANGLE_NOSE_EYES"]
            eye1 = (vec[embedding_position["EYE"]], vec[embedding_position["EYE"] + 1])
            eye2 = (vec[embedding_position["EYE"] + 2], vec[embedding_position["EYE"] + 3])
            nose = (vec[embedding_position["NOSE"]], vec[embedding_position["NOSE"] + 1])
            if eye1[0] != -1 and eye2[0] != -1 and nose[0] != -1:
                vec[pos] = angle_eval(eye1, nose, eye2)

        return vec

    def embedding4(prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (1, 2),
            "MOUTH": 3,
            "NOSE": 4,
            "ANGLE_NOSE_EYES": 5,
            "EYES VISIBILITY": 6
        }
        dictionary_eyes_nose = {"EYE": []}

        vec = np.full(7, -1.0, dtype=float)
        vec[embedding_position["EYES VISIBILITY"]] = 0
        for cls in prediction.cls:
            class_label = classes_fd[cls.item()]
            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]
                vec[pos] = 1

                dictionary_eyes_nose["EYE"].append((bbox[0], bbox[1]))
                vec[embedding_position["EYES VISIBILITY"]] += 1

            elif class_label != "BABY":
                pos = embedding_position[class_label]
                vec[pos] = 1

                if class_label == "NOSE":
                    dictionary_eyes_nose["NOSE"] = (bbox[0], bbox[1])

            pos = embedding_position["ANGLE_NOSE_EYES"]

            if len(dictionary_eyes_nose) == 2 and dictionary_eyes_nose["EYE"] == 2:
                eye1 = dictionary_eyes_nose["EYE"][0]
                eye2 = dictionary_eyes_nose["EYE"][1]
                nose = dictionary_eyes_nose["NOSE"]
                vec[pos] = angle_eval(eye1, nose, eye2)
        return vec

    def embedding5(prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (2, 4),
            "MOUTH": 6,
            "NOSE": 8
        }

        vec = np.full(10, -1.0, dtype=float)
        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = classes_fd[cls.item()]
            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]
            elif class_label != "BABY":
                pos = embedding_position[class_label]

                vec[pos] = bbox[0]
                vec[pos + 1] = bbox[1]

        return vec

    def embedding6(prediction: Boxes):
        embedding_position = {
            "HEAD": 0,
            "EYE": (1, 2),
            "MOUTH": 3,
            "NOSE": 4
        }

        vec = np.full(5, -1.0, dtype=float)
        for cls in (prediction.cls):
            class_label = classes_fd[cls.item()]
            if class_label == "EYE":
                pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                    embedding_position[class_label][1]
                vec[pos] = 1
            elif class_label != "BABY":
                pos = embedding_position[class_label]
                vec[pos] = 1
        return vec

    # funzioni di geom analitica per emebedding supremo
    def compute_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def compute_point_to_line_distance(point, line_start, line_end):
        # Distance from a point to a line defined by line_start and line_end
        return (np.abs(np.cross(np.array(line_end) - np.array(line_start),
                                np.array(line_start) - np.array(point))) / compute_distance(line_start, line_end))

    def compute_face_angle(left_eye, nose, mouth):
        # Computes the angle at nose between left_eye–nose–mouth
        vector1 = np.array(left_eye) - np.array(nose)
        vector2 = np.array(mouth) - np.array(nose)
        cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        return np.degrees(angle_rad)
'''
