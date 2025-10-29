# SIDS DETECTOR
![Demo](README_resources/introduction.gif)
<details>
<summary><span style="font-size: 2em; font-weight: 600;">📖 Table of Contents</span></summary>

- [🎯 Project Goal](#-project-goal)  
- [❓ Problem Definition](#problem-definition)  
- [💡 Proposed Solution](#-proposed-solution)  
- [🔄 System Workflow](#-system-workflow)  
- [🧠 Computer Vision Model Pipeline](#-computer-vision-model-pipeline)  
  - [🔍 YOLOv8 Face Detection](#-yolov8-face-detection-model)  
  - [🏃 YOLOv8 Pose Estimation](#-yolov8-pose-estimation-model)  
  - [🛠️ Feature Engineering and Embedding](#feature-engineering-and-embedding)
  - [📊 XGBoost Classifier](#-xgboost-classifier)  
- [🎥 Qualitative Results](#-qualitative-results)  
- [🔧 Let's set up the project locally!](#-do-you-want-to-work-on-it-locally-lets-set-up-the-project)
- [🚀 Running the Project in Google Colab](#-do-you-want-to-play-with-a-google-colab-demo-online-lets-do-it)
</details>

## 🎯 Project goal
This project aims to develop a computer vision–based **monitoring system** to track infants while they sleep and detect potentially dangerous positions that may increase **the risk of Sudden Infant Death Syndrome (SIDS)**.
## ❓Problem definition
SIDS is strongly associated with situations in which a baby’s airways become obstructed during sleep.The sleeping position plays a critical role:
- 🟥 **Prone position** (lying face down) – highest risk of suffocation. 
- 🟧 **Side positions** (lying on the left or right side) – increased risk if the baby’s mouth and nose are pressed against the mattress, pillow, or blanket.
- 🟩 **Supine position** (on the back, face up) → safest position, airways unobstructed.

**In this project, we define:**
- **Baby Safe**: infant is in a safe sleeping position.
- **Baby in Danger**: infant is in a potentially risky position.

## 💡 Proposed solution
We aim to create a smart baby monitor positioned above the crib, continuously analyzing video frames in real time.

At this stage of development, our focus is on the computer vision model, responsible for:
- Analyzing video frames in real time.
- Detecting the baby’s body pose and face features.
- Classifying frames as "baby safe" or "baby in danger".

➡️ Future work will integrate this into a real monitoring device, connected to an alert system to notify parents when a dangerous posture is detected.

## 🔄 System Workflow
Overall, the smart baby monitor works like this:
```mermaid
flowchart TD
  A[Camera above crib] --> B[Frame acquisition]
  B --> C[Preprocessing with CLAHE]
  C --> D[Computer Vision Model]
  D --> G{Inference}
  G -->|Baby in danger| H[⚠️Alert]
  G -->|Baby safe| I[✅No action]
```

## 🧠 Computer vision model pipeline
During this stage, we focused on the computer vision model, whose processing pipeline includes:
```mermaid
flowchart TD
    A[Frame] --> |YOLOv8 face detection| B[Face features]
    A--> |YOLOv8 pose estimation| C[Pose features]
    B--> |Feature elaboration:\n angles and distances evaluation, normalization| D
    C--> |Feature elaboration:\n angles and distances evaluation, normalization| D
    D[Embedding with face and pose information]
    D-->|Processing with MLP| E[ Learned embedding ]
    E--> G[XGBoost classifier]
    G--> H{Inference}
    H -->|Baby in danger| I( )
    H -->|Baby safe| K( )
    
    
```

### 🔍 YOLOv8 face detection model
The first element of our pipeline is the YOLOv8 face detection model
It was fine-tuned with [this dataset](https://app.roboflow.com/sids-project-3gvel/dataset_v3-hn7xa/1) after careful data augmentation.

### 🏃 YOLOv8 pose estimation model
The second element of our pipeline is the YOLOv8 pose estimation model.
It was fine-tuned with [this dataset](https://universe.roboflow.com/sids-project-3gvel/pose_estimation-merged-gdksv/dataset/1m) after careful data augmentation.


### 🛠️ Feature Engineering and Embedding
From YOLO outputs, we extract handcrafted features (86 total):
- **Face-related**: landmark presence, normalized positions, geometric relations (angles, aspect ratios, eye-to-mouth distances).
- **Pose-related**: normalized keypoints, distances between joints, torsion, and angles.

➡️ These features are passed through an MLP trained with Supervised Contrastive Loss, producing a 32-dimensional learned embedding.

### 📊 XGBoost classifier
- Multiple classifiers were tested with hyperparameter optimization, and the best one was selected to predict “baby safe” or “baby in danger”.
- XGBoost achieved the best balance of:
	- ✅ High recall (especially for “baby in danger” cases).
	- ✅ Strong accuracy.
	- ✅ Low overfitting tendency.

| | **precision** | **recall** | **f1-score** | **support** |
| :--- | :---: | :---: | :---: | :---: |
| **baby_safe** | 0.96 | 0.94 | 0.95 | 423 |
| **baby_unsafe** | 0.94 | 0.96 | 0.95 | 409 |
| | | | | |
| **accuracy** | | | 0.95 | 832 |
| **macro avg** | 0.95 | 0.95 | 0.95 | 832 |
| **weighted avg**| 0.95 | 0.95 | 0.95 | 832 |

<img src="README_resources/confusion_matrix.png" width="40%">

## 🎥 Qualitative results
![Demo](README_resources/classification_demo.gif)


The demo video was AI generated using VEO.

Our model is frame-based. Future work may include inter-frame analysis to stabilize predictions, but this provides a solid foundation for further refinement.

## 🔧 Do you want to work on it locally? Let's set up the project!

### Requirements
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA support recommended)
- Ultralytics YOLOv8
- XGBoost
- Scikit-learn
- OpenCV
- Numpy, Pandas, Matplotlib
### Setup
Clone the repository and install dependencies:
- **For Cuda**
    ```bash
    git clone https://github.com/Lori-in-the-clouds/SIDS_revelation_project.git
    pip install -r cuda_requirements.txt
    ```
- **For Apple Silicon (Metal)**
    ```bash
    git clone https://github.com/Lori-in-the-clouds/SIDS_revelation_project.git
    pip install -r mps_requirements.txt
    ```
  
- **For Linux without GPU**
    ```bash
    git clone https://github.com/Lori-in-the-clouds/SIDS_revelation_project.git
    conda create -n nome_ambiente python=3.10.19
    pip install -r linux_requirements.txt
    ```

### Test it running inference on an .mp4 video
To run inference on an `.mp4` video:
```bash
python main.py --input data/sample_video.mp4
```
The script generates a video with annotated bounding boxes and keypoints, saved in the same location where the input file is with the name
`namefile_pred_with_filters.mp4`, where namefile corresponds to the input filename.

### 🎬 Example Input Video
To test the model, you can use this example input video:  
🎬 [⬇️ Download sample_video.mp4](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/raw/main/video_prediction/video/baby_1.mp4)  
This short clip simulates a sleeping infant transitioning between different positions. You can replace it with your own `.mp4` file for testing.

## 🚀 Do you want to play with a Google Colab demo online? Let's do it!
Open ... with Google Colab and upload your video...just click here: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lori-in-the-clouds/SIDS_revelation_project/blob/main/colab_demo.ipynb)
