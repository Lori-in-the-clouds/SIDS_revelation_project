import numpy as np
from ultralytics.engine.results import Boxes

#funzione per calcolare l'angolo tra eye1, nose e eye2 che sono tuple di due float
def angle_eval(eye1, nose, eye2):

    eye1_nose = np.array(eye1) - np.array(nose)
    eye2_nose = np.array(eye2) - np.array(nose)

    cos_theta = np.dot(eye1_nose, eye2_nose) / (np.linalg.norm(eye1_nose) * np.linalg.norm(eye2_nose))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    angle_deg = np.degrees(angle_rad)
    return angle_deg


def embedding1(prediction: Boxes):
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
        for cls in (prediction.cls):
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

            if len(dictionary_eyes_nose) == 2 and dictionary_eyes_nose["EYE"]==2:
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
            vec[embedding_position["EYES VISIBILITY"]]+= 1
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
    for cls in (prediction.cls):
        class_label = classes_fd[cls.item()]
        if class_label == "EYE":
            pos = embedding_position[class_label][0] if vec[embedding_position[class_label][0]] == -1 else \
                embedding_position[class_label][1]
            vec[pos] = 1

            dictionary_eyes_nose["EYE"].append((bbox[0], bbox[1]))
            vec[embedding_position["EYES VISIBILITY"]]+=1

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
            elif class_label !="BABY":
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


#funzioni di geom analitica per emebedding supremo
def compute_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compute_point_to_line_distance(point, line_start, line_end):
    # Distance from a point to a line defined by line_start and line_end
    return np.abs(np.cross(np.array(line_end) - np.array(line_start), np.array(line_start) - np.array(point))) / compute_distance(line_start, line_end)

def compute_face_angle(left_eye, nose, mouth):
    # Computes the angle at nose between left_eye–nose–mouth
    vector1 = np.array(left_eye) - np.array(nose)
    vector2 = np.array(mouth) - np.array(nose)
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def embedding_SUPREMO(prediction: Boxes):
    keypoints = {}
    for bbox, cls in zip(prediction.xywhn, prediction.cls):
        class_label = classes_fd[cls.item()]
        match class_label:
            case "EYE":
                if "eye1" in keypoints:
                    keypoints["eye2"] = (bbox[0], bbox[1])
                else:
                    keypoints["eye1"] = (bbox[0], bbox[1])
            case "MOUTH":
                keypoints["mouth"] = (bbox[0], bbox[1])
            case "NOSE":
                keypoints["nose"] = (bbox[0], bbox[1])

    presence_flags = [
        int('eye1' in keypoints),
        int('eye2' in keypoints),
        int('nose' in keypoints),
        int('mouth' in keypoints),
    ]
    eye1= keypoints["eye1"]
    eye2 = keypoints["eye2"]
    nose = keypoints["nose"]
    mouth = keypoints["mouth"]

    coordinates = list(eye1) + list(eye2) + list(nose) + list(mouth)
    # 13: distance between eyes
    eye_distance = compute_distance(eye1, eye2)

    # 14: vertical face length (nose to mouth)
    face_vertical_length = compute_distance(nose, mouth)

    # 15: angle between eye1 – nose – mouth
    face_angle_vertical = compute_face_angle(eye1, nose, mouth)

    #angle between eye1-nose-eye2
    face_angle_horizontal = compute_face_angle(eye1, nose, eye2)

    # 16: symmetry (difference of eye distances to nose–mouth line)
    try:
        eye1_to_axis = compute_point_to_line_distance(eye1, nose, mouth)
        eye2_to_axis = compute_point_to_line_distance(eye2, nose, mouth)
        symmetry_diff = abs(eye1_to_axis - eye2_to_axis)
    except:
        symmetry_diff = 0.0
# Final embedding
    embedding = (
        presence_flags +
        coordinates +
        [eye_distance, face_vertical_length, face_angle, symmetry_diff]
    )

    return np.array(embedding)