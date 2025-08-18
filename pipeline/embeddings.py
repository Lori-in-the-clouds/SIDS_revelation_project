import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics import YOLO

from pathlib import Path
import os
import json
import cv2
from datetime import datetime

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

''' EMBEDDING BUILDER CLASS '''


class EmbeddingBuilder:
    def __init__(self, weights_path: str, dataset_path: str, mode: str):
        """
        Initialize the EmbeddingBuilder.

        After initialization, the following information is available:
        - Dataset info (self.file_label, self.dim_dataset, self.classes_bs)
        - Extracted features (self.features)
          A dictionary of features with coordinates:
                - 'eye1', 'eye2' : tuples (x, y) for eyes
                - 'nose'          : tuple (x, y) for nose
                - 'mouth'         : tuple (x, y) for mouth
                - 'image_path'    : str
                - 'label':        : 1/2

        Args:
            weights_path (str): Path to YOLOv8 weights.
            dataset_path (str): Path to the back/stomach dataset.
            mode (str): Operation mode.
                - "extract_features":
                  Extracts features from each image and saves them in a `.npy` file
                  inside the dataset folder.
                - "extract_features_imageswithinference":
                  Same as above, but also saves images with inference results in a
                  dedicated folder inside the model folder.
                - "load":
                  Loads features from a `.npy` file inside the dataset folder.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        # load YOLOv8 model
        self.model_fd = YOLO(weights_path)
        self.classes_fd = self.model_fd.names
        self.model_version = weights_path.split(".weights")[0][-1]

        # dataset info
        self.dataset = Path(dataset_path)
        self.file_label = {}
        self.dim_dataset = 0
        self.classes_bs = None

        self.features = []
        self.y = []
        self.image_paths=[]

        if mode == "extract_features":
            self.process_dataset("default")
        elif mode == "extract_features_imageswithinference":
            self.process_dataset("imageswithinference")
        elif mode == "load":
            self.extract_dataset_info()
            self.load_features_and_y()
        else:
            raise ValueError(
                f"Invalid mode '{mode}\n Expected one of: 'extract_features', 'extract_features_imageswithinference', 'load'")
        print("")
        print("Embedding builder initialized successfully".ljust(90, '-'))
        print(f"Face detection model: {self.model_version} (YOLOv8)")
        print(f"Dataset: {self.dataset}")
        print(f"Dataset dimension: {self.dim_dataset}")
        print(f"Dataset labels: {self.classes_bs}")
        print("".ljust(90, '-'))

    def progress_debug(self, var_to_monitor: list):
        """
        To log precessing of dataset's images
        """
        if len(var_to_monitor) % 100 == 0:
            print(
                f"{int(len(var_to_monitor) * 100 / self.dim_dataset)}%-->    {len(var_to_monitor)} / {self.dim_dataset} files processed")

    def extract_dataset_info(self):
        """
        Given a dataset in coco.json format, extract:
            - self.file_label: {file_name -> label_id}
            - self.classes_bs: {class_name -> class_id}
            - self.dim_dataset: number of valid labeled images
        """
        print("")
        print("Extracting dataset info from .coco.json file:".ljust(90, '-'))
        with open(f"{str(self.dataset)}/_annotations.coco.json", "r") as f:
            dataset = json.load(f)

            file_imgid = {img["file_name"]: img["id"] for img in dataset["images"]}
            imgid_label = {ann["image_id"]: ann["category_id"] for ann in dataset["annotations"]}

            classes_bs = {cls["name"]: cls["id"] for cls in dataset["categories"]}
            del classes_bs["baby"]
            del classes_bs["crib"]

        file_label = {file: imgid_label[img_id] for file, img_id in file_imgid.items() if
                      img_id in imgid_label and imgid_label[img_id] in classes_bs.values()}

        self.dim_dataset = len(file_label)
        self.classes_bs = classes_bs
        self.file_label = file_label
        print(f"Dataset contains {self.dim_dataset} valid samples, and labels are {self.classes_bs}")
        print("".ljust(90, '-'))

    def features_extractor(self, prediction: Boxes):
        """
        Extract features from YOLO prediction boxes.

        Parameters:
            prediction (Boxes): YOLO prediction containing bounding boxes and class IDs.

        Returns:
            A dictionary of features with coordinates:
                - 'eye1', 'eye2' : tuples (x, y) for eyes
                - 'nose'          : tuple (x, y) for nose
                - 'mouth'         : tuple (x, y) for mouth
                - 'image_path'    : str
                - 'label':        : 1/2

        Notes:
            - Coordinates are normalized values in [0, 1] range as provided by YOLO output.
        """
        ft = {}

        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = self.classes_fd[cls.item()].lower()

            if class_label == "eye":
                if "eye1" in ft:
                    ft["eye2"] = (bbox[0], bbox[1])
                else:
                    ft["eye1"] = (bbox[0], bbox[1])
            elif class_label == "mouth":
                ft["mouth"] = (bbox[0], bbox[1])
            elif class_label == "nose":
                ft["nose"] = (bbox[0], bbox[1])
        return ft

    def save_features_and_y(self):
        """
            Save extracted features and corresponding labels to .npy files.
        """
        print("")
        print("Saving features in .npy:".ljust(90, '-'))
        np.save(f"{str(self.dataset)}/baseline_model{self.model_version}_features.npy", self.features)
        np.save(f"{str(self.dataset)}/baseline_model{self.model_version}_labels.npy", self.y)

        print(f"Features saved in '{str(self.dataset)}/baseline_model{self.model_version}_features.npy' and labels saved in '{str(self.dataset)}/baseline_model{self.model_version}_labels.npy")
        print("".ljust(90, '-'))

    def load_features_and_y(self):
        """
            Load features and corresponding labels from .npy files.
        """
        print("")
        print("Loading features from .npy".ljust(90, '-'))
        self.features = np.load(f"{str(self.dataset)}/baseline_model{self.model_version}_features.npy",
                                allow_pickle=True).tolist()
        self.y = np.load(f"{str(self.dataset)}/baseline_model{self.model_version}_labels.npy",
                         allow_pickle=True).tolist()
        self.dim_dataset = len(self.y)

        print(
            f"features and labels loaded succesfully, in particular there are {self.dim_dataset} files in the dataset")
        print("".ljust(90, '-'))

    def process_dataset(self, mode: str):
        """
        Extract features and labels from all `.jpg` images in the dataset.

        - Loads dataset info and maps images to labels.
        - Runs detection with `self.model_fd` and extracts features and labels
        - If `mode == "imageswithinference"`, saves images with drawn bounding boxes
          to a dedicated 'prediction' folder in the model folder.
        - Saves extracted features and labels in a file .npy.

        Parameters
        ----------
        mode : str
            - "default": extract features and labels and save them
            - "imageswithinference": same as above + save images with face detection inference (bboxes)
        """
        # extract classes_bs, dim_dataset, file_label dictionary
        self.extract_dataset_info()

        # prepare output_dir for imageswithinference
        output_dir = None
        if mode == "imageswithinference":
            date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
            output_dir = Path(f"../models/{self.model_version}.predictions.{date_str}")
            output_dir.mkdir(exist_ok=True, parents=True)

        print("")
        print("Feature extraction for each dataset image".ljust(90, '-') if mode == "default"
              else "Feature extraction + generation of image with bboxes for each dataset image".ljust(90, '-'))
        # extract features from each file
        for img_path in self.dataset.glob("*.jpg"):
            if img_path.name not in self.file_label:
                continue

            self.progress_debug(self.y)
            result = self.model_fd(img_path, conf=0.3, verbose=False)[0]

            ft = self.features_extractor(result.boxes)
            ft["file_path"] = img_path.name

            label = self.file_label[img_path.name]
            ft["label"] = label

            self.features.append(ft)
            self.y.append(label)
            self.image_paths.append(img_path.name)

            if output_dir:
                # save image with bboxes
                img = cv2.imread(str(img_path))
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(output_dir / img_path.name), img)

        print(f"{len(self.y)} image processed, features(self.features) and labels(self.y) extracted")
        print("".ljust(90, '-'))

        # save features and labels in .npy files
        self.save_features_and_y()

    def embedding_positions(self):
        """
          Create embeddings from facial keypoints.

          Returns
          -------
          tuple[list[list[float]], list[str]]
              A tuple containing:
              - X: list of embedding vectors
              - features_names: list of feature names
        """
        print("")
        print("Creation of positions features embedding".ljust(90, '-'))

        X = []
        features_names = [ "x_eye1", "y_eye1", "x_eye2", "y_eye2",
                    "x_nose", "y_nose", "x_mouth", "y_mouth"]

        for ft, cls in zip(self.features, self.y):
            self.progress_debug(X)
            presence_flags = [
                int('eye1' in ft),
                int('eye2' in ft),
                int('nose' in ft),
                int('mouth' in ft),
            ]
            eye1 = ft["eye1"] if presence_flags[0] == 1 else (-1, -1)
            eye2 = ft["eye2"] if presence_flags[1] == 1 else (-1, -1)
            nose = ft["nose"] if presence_flags[2] == 1 else (-1, -1)
            mouth = ft["mouth"] if presence_flags[3] == 1 else (-1, -1)

            # coordinates features
            coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)

            # final embedding
            embedding = coordinates
            X.append(embedding)

        print(f"{len(X)} embedding created")
        print("".ljust(90, '-'))

        return X, features_names

    def embedding_flags(self):
        """
          Create embeddings from facial keypoints.

          Returns
          -------
          tuple[list[list[float]], list[str]]
              A tuple containing:
              - X: list of embedding vectors
              - features_names: list of feature names
        """
        print("")
        print("Creation of flags features embedding".ljust(90, '-'))

        X = []
        features_names = ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth"]

        for ft, cls in zip(self.features, self.y):
            self.progress_debug(X)
            presence_flags = [
                int('eye1' in ft),
                int('eye2' in ft),
                int('nose' in ft),
                int('mouth' in ft),
            ]
            # final embedding
            embedding = presence_flags
            X.append(embedding)

        print(f"{len(X)} embedding created")
        print("".ljust(90, '-'))

        return X, features_names

    def embedding_all_features(self):
        """
          Create embeddings from facial keypoints.

          Returns
          -------
          tuple[list[list[float]], list[str]]
              A tuple containing:
              - X: list of embedding vectors
              - features_names: list of feature names
        """
        print("")
        print("Creation of all features embedding".ljust(90, '-'))

        X = []
        features_names = ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth", "x_eye1", "y_eye1", "x_eye2", "y_eye2",
                    "x_nose", "y_nose", "x_mouth", "y_mouth", "eye_distance", "face_vertical_length",
                    "face_angle_vertical", "face_angle_horizontal", "symmetry_diff"]

        for ft, cls in zip(self.features, self.y):
            self.progress_debug(X)
            presence_flags = [
                int('eye1' in ft),
                int('eye2' in ft),
                int('nose' in ft),
                int('mouth' in ft),
            ]
            eye1 = ft["eye1"] if presence_flags[0] == 1 else (-1, -1)
            eye2 = ft["eye2"] if presence_flags[1] == 1 else (-1, -1)
            nose = ft["nose"] if presence_flags[2] == 1 else (-1, -1)
            mouth = ft["mouth"] if presence_flags[3] == 1 else (-1, -1)

            # coordinates features
            coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)

            # distance between eyes
            eye_distance = compute_distance(eye1, eye2) if (presence_flags[0] * presence_flags[1]) == 1 else -1

            # vertical face length (nose to mouth)
            face_vertical_length = compute_distance(nose, mouth) if (presence_flags[2] * presence_flags[3]) == 1 else -1

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
                    [eye_distance, face_vertical_length, face_angle_vertical, face_angle_horizontal, symmetry_diff]
            )
            X.append(embedding)

        print(f"{len(X)} embedding created")
        print("".ljust(90, '-'))

        return X, features_names

    def embedding_all_features_norm(self):
        print("")
        print("Creation of all features embedding".ljust(90, '-'))

        X = []
        features_name = ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth",
                    "x_eye1", "y_eye1", "x_eye2", "y_eye2", "x_nose", "y_nose", "x_mouth", "y_mouth",
                    "x_eye1_norm", "y_eye1_norm", "x_eye2_norm", "y_eye2_norm", "x_nose_norm", "y_nose_norm",
                    "x_mouth_norm", "y_mouth_norm",
                    "eye_distance", "eye_distance_norm", "face_vertical_length", "face_vertical_length_norm",
                    "face_angle_vertical", "face_angle_horizontal", "symmetry_diff", "head_ration"]

        for kpt, cls in zip(self.features, self.y):
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

            coordinates_norm = normalize(eye1, head) + normalize(eye2, head) + normalize(nose, head) + normalize(mouth,
                                                                                                                 head)

            # head h/w ration
            head_ration = (head[3] / head[2]) if (head != -1) else -1

            # distance between eyes
            eye_distance = compute_distance(eye1, eye2) if (presence_flags[0] * presence_flags[1]) == 1 else -1
            eye_distance_norm = (eye_distance / head[2]) if (eye_distance != -1 and head != -1) else -1

            # vertical face length (nose to mouth)
            face_vertical_length = compute_distance(nose, mouth) if (presence_flags[2] * presence_flags[
                3]) == 1 else -1
            face_vertical_length_norm = (face_vertical_length / head[3]) if (
                        face_vertical_length != -1 and head != -1) else -1

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
                    [eye_distance, eye_distance_norm, face_vertical_length, face_vertical_length_norm,
                     face_angle_vertical, face_angle_horizontal, symmetry_diff, head_ration]
            )
            X.append(embedding)

        print(f"FINISHED: {len(X)} embedding created")
        return X, features_name
