import numpy as np
import ultralytics
from pandas import DataFrame
from ultralytics.engine.results import Boxes
from ultralytics import YOLO
from pathlib import Path
import json
import cv2
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler

def standard_scaler_embeddings(embeddings: DataFrame):
    embeddings_df = embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_df)
    embeddings_scaled_df = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns, index=embeddings_df.index)
    return embeddings_scaled_df

''' GEOMETRIC FUNCTIONS '''

def compute_distance(point1, point2):
    if point1 == (-1, -1) or point2 == (-1, -1):
        return -1
    return np.linalg.norm(np.array(point1) - np.array(point2))


def compute_point_to_line_distance(point, line_start, line_end):
    # Distance from a point to a line defined by line_start and line_end
    return (np.abs(np.cross(np.array(line_end) - np.array(line_start),
                            np.array(line_start) - np.array(point))) / compute_distance(line_start, line_end))


def compute_face_angle(el1, nose, el2):
    if el1 == (-1, -1) or nose == (-1, -1) or el2 == (-1, -1):
        return -1
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
        return [a / b for a, b in zip(coordinates, head_xy)]



''' EMBEDDING BUILDER CLASS '''
class EmbeddingBuilder:
    def __init__(self, weights_path_fd: str, dataset_path: str, mode: str, weights_path_pe: str=None):
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
        self.model_fd = YOLO(weights_path_fd)
        self.classes_fd = self.model_fd.names
        self.model_version_fd = weights_path_fd.split(".fd_weights")[0][-1]
        self.model_pe = YOLO(weights_path_pe) if weights_path_pe else None
        self.model_version_pe=weights_path_pe.split(".pe_weights")[0][-1] if weights_path_pe else None

        # dataset info
        self.dataset = Path(dataset_path)
        self.file_label = {}
        self.dim_dataset = 0
        self.classes_bs = None

        self.features = []
        self.y = []
        self.image_paths = []

        if mode == "extract_features":
            self.process_dataset("default")
        elif mode == "extract_features_imageswithinference":
            self.process_dataset("imageswithinference")
        elif mode == "load":
            self.extract_dataset_info()
            self.load_features()
        else:
            raise ValueError(
                f"Invalid mode '{mode}\n Expected one of: 'extract_features', 'extract_features_imageswithinference', 'load'")
        self.normalize_labels()
        self.update_classes()

        print("")
        print("Embedding builder initialized successfully".ljust(90, '-'))
        print(f"Face detection model: {self.model_version_fd} (YOLOv8)")
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
        ft = {
            "eye1": (-1, -1),
            "eye2": (-1, -1),
            "nose": (-1, -1),
            "mouth": (-1, -1),
            "head": (-1, -1)
        }

        for bbox, cls in zip(prediction.xywhn, prediction.cls):
            class_label = self.classes_fd[cls.item()].lower()
            x = float(bbox[0].item())
            y = float(bbox[1].item())

            w = float(bbox[2].item())
            h = float(bbox[3].item())
            if class_label == "eye":
                if ft["eye1"] == (-1, -1):
                    ft["eye1"] = (x, y)
                else:
                    ft["eye2"] = (x, y)
            elif class_label == "mouth":
                ft["mouth"] = (x, y)
            elif class_label == "nose":
                ft["nose"] = (x, y)
            elif class_label == "head":
                ft["head"] = (x, y, w, h)
        return ft

    def features_extractor_keypoints(self, prediction:ultralytics.engine.results.Results):
        keypoint_names = [
            "nose_k", "left_eye_k", "right_eye_k", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        if prediction.keypoints.conf.shape[0] == 0:
            return {name: (-1, -1) for name in keypoint_names}

        else:
            conf = prediction.keypoints.conf.numpy()[0].reshape(-1, 1).T
            #data_xy = prediction.keypoints.xy.numpy()[0].T
            data_xyn = prediction.keypoints.xyn.numpy()[0].T

            merged = np.vstack((data_xyn, conf))
            keypoints = pd.DataFrame(merged, index=["x", "y", "conf"], columns=keypoint_names)
            kpt = {}
            for col in keypoints.columns:
                if keypoints.loc["conf", col] < 0.1:
                    kpt[col] = (-1, -1)
                else:
                    x = float(keypoints.loc["x", col])
                    y = float(keypoints.loc["y", col])
                    kpt[col] = (x, y)
            return kpt



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
            output_dir = Path(f"../datasets/fd_prediction/{self.dataset.name}")
            output_dir.mkdir(exist_ok=True, parents=True)

        print("")
        print("Feature extraction for each dataset image".ljust(90, '-') if mode == "default"
              else "Feature extraction + generation of image with bboxes for each dataset image".ljust(90, '-'))
        # extract features from each file
        for img_path in self.dataset.glob("*.jpg"):
            if img_path.name not in self.file_label:
                continue

            self.progress_debug(self.features)
            result_fd = self.model_fd(img_path, conf=0.3, verbose=False)[0]
            result_pe =self.model_pe(img_path, conf=0.3, verbose=False)[0] if self.model_pe else None

            ft_fd = self.features_extractor(result_fd.boxes)
            ft_pe = self.features_extractor_keypoints(result_pe) if self.model_pe else {}

            ft = ft_pe | ft_fd
            ft["image_path"] = img_path.name

            label = self.file_label[img_path.name]
            ft["label"] = label

            self.features.append(ft)

            if output_dir and (len(self.features) < self.dim_dataset/100):
                if self.model_pe:
                    # keypoints (BGR ndarray)
                    img_with_kp = result_pe.plot()

                    # bboxes
                    for box in result_fd.boxes :
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(img_with_kp, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.imwrite(str(output_dir / img_path.name), img_with_kp)

                else:
                    # save image with bboxes
                    img = cv2.imread(str(img_path))
                    for box in result_fd.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(str(output_dir / img_path.name), img)

        self.features = pd.DataFrame(self.features)
        self.y = self.features["label"].tolist()
        self.image_paths = self.features["image_path"].tolist()
        print(f"{len(self.y)} image processed, features(self.features) and labels(self.y) extracted")
        print("".ljust(90, '-'))

        # save features and labels in .csv files
        self.save_features()

    def normalize_labels(self):
        if min(set(self.features["label"])) != 0:
            self.features["label"] -= 1

        if min(set(self.classes_bs.values())) != 0:
            self.classes_bs = {entry[0]: (entry[1] - 1) for entry in self.classes_bs.items()}

        if min(set(self.y)) != 0:
            self.y = np.array(self.y) - 1

    def update_classes(self):
        self.classes_bs = {"baby_safe" if entry[0] == "baby_on_back" else "baby_unsafe": (entry[1]) for entry in
                           self.classes_bs.items()}

    def save_features(self):
        """
                    Save extracted features in .csv file in the dataset folder.
                """
        print("")
        print(
            "Saving features in .csv\nDataframe with columns [eye1, eye2, nose, mouth, head, label, image_path]. [eye1, eye2, mouth, label, head] are equal to (-1, -1) if were not detected:".ljust(
                90, '-'))
        self.features.to_csv(f"{str(self.dataset)}/model{self.model_version_fd}_features{'_keypoints' if self.model_pe else ''}{self.model_version_pe if self.model_version_pe else ''}.csv", index=False)

        print(
            f"Features saved in '{str(self.dataset)}/model{self.model_version_fd}_features{'_keypoints' if self.model_pe else ''}{self.model_version_pe if self.model_version_pe else ''}.csv'")
        print("".ljust(90, '-'))

    def load_features(self):
        """
            Load features from .csv file in the dataset folder and populate self.y,  self.image_paths, self.dim_dataset.
        """
        print("")
        print("Loading features from .csv".ljust(90, '-'))
        self.features = pd.read_csv(f"{str(self.dataset)}/model{self.model_version_fd}_features{'_keypoints' if self.model_pe else ''}{self.model_version_pe if self.model_version_pe else ''}.csv")
        exclude_cols = {"label", "image_path"}
        for col in self.features.columns:
            if col not in exclude_cols:
                self.features[col] = self.features[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

        self.y = self.features["label"].tolist()
        self.image_paths = self.features["image_path"].tolist()
        self.dim_dataset = self.features.shape[0]

        print(
            f"Features loaded succesfully, in particular there are {self.dim_dataset} files in the dataset")
        print("".ljust(90, '-'))

    def extract_flags(self, ft: dict):
        presence_flags = [
            int(ft.get("eye1", (-1, -1)) != (-1, -1)),
            int(ft.get("eye2", (-1, -1)) != (-1, -1)),
            int(ft.get("nose", (-1, -1)) != (-1, -1)),
            int(ft.get("mouth", (-1, -1)) != (-1, -1)),
        ]
        return presence_flags

    def extract_coordinates(self, ft: dict):
        eye1 = ft["eye1"]
        eye2 = ft["eye2"]
        nose = ft["nose"]
        mouth = ft["mouth"]

        coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)
        return coordinates

    def extract_normalized_coordiates(self, ft: dict):
        eye1 = ft["eye1"]
        eye2 = ft["eye2"]
        nose = ft["nose"]
        mouth = ft["mouth"]
        head = ft["head"] if ft["head"] != (-1, -1) else -1

        coordinates_norm = normalize(eye1, head) + normalize(eye2, head) + normalize(nose, head) + normalize(mouth,
                                                                                                             head)
        return coordinates_norm

    def extract_geometric_info(self, ft: dict):
        presence_flags = [
            int(ft["eye1"] != (-1, -1)),
            int(ft["eye2"] != (-1, -1)),
            int(ft["nose"] != (-1, -1)),
            int(ft["mouth"] != (-1, -1)),
        ]

        eye1 = ft["eye1"]
        eye2 = ft["eye2"]
        nose = ft["nose"]
        mouth = ft["mouth"]
        head = ft["head"] if ft["head"] != (-1, -1) else -1

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

        geometric_info = [eye_distance, eye_distance_norm, face_vertical_length, face_vertical_length_norm,
                     face_angle_vertical, face_angle_horizontal, symmetry_diff, head_ration]
        return geometric_info

    """     Embeddings types        """
    def create_embedding(self, flags=False, positions=False, positions_normalized=False, geometric_info=False, k_positions_normalized=False, k_geometric_info=False):
        features_names = []
        if flags:
            features_names += ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth"]
        if positions:
            features_names += ["x_eye1", "y_eye1", "x_eye2", "y_eye2", "x_nose", "y_nose", "x_mouth", "y_mouth"]
        if positions_normalized:
            features_names += ["x_eye1_norm", "y_eye1_norm", "x_eye2_norm", "y_eye2_norm", "x_nose_norm", "y_nose_norm",
                               "x_mouth_norm", "y_mouth_norm"]
        if geometric_info:
            features_names += ["eye_distance", "eye_distance_norm", "face_vertical_length", "face_vertical_length_norm",
                               "face_angle_vertical", "face_angle_horizontal", "symmetry_diff", "head_ration"]
        if k_positions_normalized:
            features_names += [ "x_nose_k", "y_nose_k", "x_left_eye_k", "y_left_eye_k", "x_right_eye_k", "y_right_eye_k", "x_left_ear", "y_left_ear", "x_right_ear","y_right_ear",
                                "x_left_shoulder","y_left_shoulder", "x_right_shoulder", "y_right_shoulder", "x_left_elbow","y_left_elbow", "x_right_elbow","y_right_elbow",
                                "x_left_wrist","y_left_wrist", "x_right_wrist", "y_right_wrist", "x_left_hip","y_left_hip", "x_right_hip","y_right_hip",
                                "x_left_knee", "y_left_knee","x_right_knee","y_right_knee", "x_left_ankle","y_left_ankle", "x_right_ankle","y_right_ankle"
                                ]
        if k_geometric_info:
            features_names += ["shoulders_dist", "shoulder_hip_right_dist", "shoulder_hip_left_dist", "nose_shoulder_right", "nose_shoulder_left", "shoulder_left_knee_right", "shoulder_right_knee_left", "knee_ankle_right", "knee_ankle_left","nose_hip_right", "nose_hip_left"]

            features_names+= ["elbow_shoulder_hip_right","elbow_shoulder_hip_left","shoulder_elbow_wrist_right","shoulder_elbow_wrist_left",
                              "shoulder_hip_knee_right","shoulder_hip_knee_left","hip_knee_ankle_right","hip_knee_ankle_left",
                              "shoulders_line_inclination","hips_line_inclination","torsion"]


        X = []

        print("")
        print("Embedding creation".ljust(90, '-'))
        print(f"Features: {features_names}")

        for ft in self.features.to_dict(orient='records'):
            embedding = []
            if flags:
                embedding += self.extract_flags(ft)
            if positions:
                embedding += self.extract_coordinates(ft)
            if positions_normalized:
                embedding += self.extract_normalized_coordiates(ft)
            if geometric_info:
                embedding += self.extract_geometric_info(ft)
            if k_positions_normalized:
                embedding += self.normalize_wrt_body_center(ft)
            if k_geometric_info:
                embedding += self.distances_between_keypoints(ft)
                embedding += self.angles_between_keypoints(ft)

            X.append(embedding)

        print(f"FINISHED: {len(X)} embedding created")
        print("".ljust(90, '-'))

        embeddings = standard_scaler_embeddings(pd.DataFrame(X, columns=features_names))
        return embeddings

    def normalize_wrt_body_center(self, ft):

        kpts = ["nose_k", "left_eye_k", "right_eye_k", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]

        if "left_shoulder" not in ft or "right_shoulder" not in ft or ft["left_shoulder"] == (-1, -1) or ft["right_shoulder"] == (-1, -1):
            return [0.0] * (len(kpts) * 2)

        x_center = (ft["left_shoulder"][0] + ft["right_shoulder"][0]) / 2.0
        y_center = (ft["left_shoulder"][1] + ft["right_shoulder"][1]) / 2.0

        embedding = []

        for el in kpts:
            if el in ft and ft[el] != (-1, -1):
                ft[el] = (ft[el][0]-x_center, ft[el][1]-y_center) if ft[el] != (-1, -1) else ft[el]
                embedding.append(ft[el][0])
                embedding.append(ft[el][1])
            else:
                embedding.extend([0.0, 0.0])
        return embedding



    def distances_between_keypoints(self,ft):

        def safe_distance(a, b):
            if a == (-1, -1) or b == (-1, -1):
                return 0.0
            return compute_distance(a, b)

        shoulders_dist= safe_distance(ft["left_shoulder"], ft["right_shoulder"])
        shoulder_hip_right_dist = safe_distance(ft["right_shoulder"], ft["right_hip"])
        shoulder_hip_left_dist = safe_distance(ft["left_shoulder"], ft["left_hip"])
        nose_shoulder_right= safe_distance(ft["nose"], ft["right_shoulder"])
        nose_shoulder_left= safe_distance(ft["nose"], ft["left_shoulder"])
        shoulder_left_knee_right = safe_distance(ft["left_shoulder"], ft["right_knee"])
        shoulder_right_knee_left = safe_distance(ft["right_shoulder"], ft["left_knee"])
        knee_ankle_right = safe_distance(ft["right_knee"], ft["right_ankle"])
        knee_ankle_left = safe_distance(ft["left_knee"], ft["left_ankle"])
        nose_hip_right = safe_distance(ft["nose"], ft["right_hip"])
        nose_hip_left = safe_distance(ft["nose"], ft["left_hip"])


        embedding = [shoulders_dist, shoulder_hip_right_dist, shoulder_hip_left_dist, nose_shoulder_right, nose_shoulder_left, shoulder_left_knee_right, shoulder_right_knee_left, knee_ankle_right, knee_ankle_left,nose_hip_right, nose_hip_left]
        return embedding

    def angles_between_keypoints(self, ft):

        elbow_shoulder_hip_right =(compute_face_angle(ft["right_elbow"], ft["right_shoulder"], ft["right_hip"]) + 180) % 360 - 180
        elbow_shoulder_hip_left = (compute_face_angle(ft["left_elbow"], ft["left_shoulder"], ft["left_hip"]) + 180) % 360 - 180

        shoulder_elbow_wrist_right = (compute_face_angle(ft["right_shoulder"],ft["right_elbow"], ft["right_wrist"]) + 180) % 360 - 180
        shoulder_elbow_wrist_left = (compute_face_angle(ft["left_shoulder"], ft["left_elbow"], ft["left_wrist"]) + 180) % 360 - 180

        shoulder_hip_knee_right = (compute_face_angle(ft["right_shoulder"],ft["right_hip"], ft["right_knee"]) + 180) % 360 - 180
        shoulder_hip_knee_left = (compute_face_angle(ft["left_shoulder"],ft["left_hip"], ft["left_knee"]) + 180) % 360 - 180

        hip_knee_ankle_right = (compute_face_angle(ft["right_hip"], ft["right_knee"], ft["right_ankle"]) + 180) % 360 - 180
        hip_knee_ankle_left = (compute_face_angle(ft["left_hip"], ft["left_knee"], ft["left_ankle"]) + 180) % 360 - 180

        if ft.get("right_shoulder", (-1, -1)) != (-1, -1) and ft.get("left_shoulder", (-1, -1)) != (-1, -1):
            angle_shoulders = np.arctan2(ft["right_shoulder"][1] - ft["left_shoulder"][1],
                                         ft["right_shoulder"][0] - ft["left_shoulder"][0])
            shoulders_line_inclination = np.degrees(angle_shoulders)
        else:
            shoulders_line_inclination = -1#

        if ft.get("right_hip", (-1, -1)) != (-1, -1) and ft.get("left_hip", (-1, -1)) != (-1, -1):
            angle_hips = np.arctan2(ft["right_hip"][1] - ft["left_hip"][1], ft["right_hip"][0] - ft["left_hip"][0])
            hips_line_inclination = np.degrees(angle_hips)
            hips_line_inclination = (hips_line_inclination + 180) % 360 - 180
        else:
            hips_line_inclination = -1

        torsion = np.abs(shoulders_line_inclination - hips_line_inclination) if shoulders_line_inclination != -1 and hips_line_inclination != -1 else -1
        torsion = (torsion + 180) % 360 - 180

        embedding = [elbow_shoulder_hip_right, elbow_shoulder_hip_left, shoulder_elbow_wrist_right,
                     shoulder_elbow_wrist_left, shoulder_hip_knee_right, shoulder_hip_knee_left,
                     hip_knee_ankle_right, hip_knee_ankle_left,shoulders_line_inclination,hips_line_inclination,torsion]
        return embedding


    def create_embedding_for_video(self, ft: dict, flags=False, positions=False, positions_normalized=False, geometric_info=False, k_positions_normalized=False,k_geometric_info=False ):
        features_names = []
        if flags:
            features_names += ["flag_eye1", "flag_eye2", "flag_nose", "flag_mouth"]
        if positions:
            features_names += ["x_eye1", "y_eye1", "x_eye2", "y_eye2", "x_nose", "y_nose", "x_mouth", "y_mouth"]
        if positions_normalized:
            features_names += ["x_eye1_norm", "y_eye1_norm", "x_eye2_norm", "y_eye2_norm", "x_nose_norm", "y_nose_norm",
                               "x_mouth_norm", "y_mouth_norm"]
        if geometric_info:
            features_names += ["eye_distance", "eye_distance_norm", "face_vertical_length", "face_vertical_length_norm",
                               "face_angle_vertical", "face_angle_horizontal", "symmetry_diff", "head_ration"]

        if k_positions_normalized:
            features_names += [ "x_nose_k", "y_nose_k", "x_left_eye_k", "y_left_eye_k", "x_right_eye_k", "y_right_eye_k", "x_left_ear", "y_left_ear", "x_right_ear","y_right_ear",
                                "x_left_shoulder","y_left_shoulder", "x_right_shoulder", "y_right_shoulder", "x_left_elbow","y_left_elbow", "x_right_elbow","y_right_elbow",
                                "x_left_wrist","y_left_wrist", "x_right_wrist", "y_right_wrist", "x_left_hip","y_left_hip", "x_right_hip","y_right_hip",
                                "x_left_knee", "y_left_knee","x_right_knee","y_right_knee", "x_left_ankle","y_left_ankle", "x_right_ankle","y_right_ankle"
                                ]
        if k_geometric_info:
            features_names += ["shoulders_dist", "shoulder_hip_right_dist", "shoulder_hip_left_dist", "nose_shoulder_right", "nose_shoulder_left", "shoulder_left_knee_right", "shoulder_right_knee_left", "knee_ankle_right", "knee_ankle_left","nose_hip_right", "nose_hip_left"]

            features_names+= ["elbow_shoulder_hip_right","elbow_shoulder_hip_left","shoulder_elbow_wrist_right","shoulder_elbow_wrist_left",
                              "shoulder_hip_knee_right","shoulder_hip_knee_left","hip_knee_ankle_right","hip_knee_ankle_left",
                              "shoulders_line_inclination","hips_line_inclination","torsion"]

        embedding = []
        if flags:
            embedding += self.extract_flags(ft)
        if positions:
            embedding += self.extract_coordinates(ft)
        if positions_normalized:
            embedding += self.extract_normalized_coordiates(ft)
        if geometric_info:
            embedding += self.extract_geometric_info(ft)
        if k_positions_normalized:
            embedding += self.normalize_wrt_body_center(ft)
        if k_geometric_info:
            embedding += self.distances_between_keypoints(ft)
            embedding += self.angles_between_keypoints(ft)

        return np.array(embedding)







