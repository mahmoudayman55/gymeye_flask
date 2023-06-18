#!/usr/bin/env python
# coding: utf-8

# In[9]:


import mediapipe as mp
import cv2
import numpy as np
import math
import pandas as pd

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[10]:


IMPORTANT_LMS = [
"NOSE",
"LEFT_SHOULDER",
"RIGHT_SHOULDER",
"LEFT_HIP",
"RIGHT_HIP",
"LEFT_KNEE",
"RIGHT_KNEE",
"LEFT_ANKLE",
"RIGHT_ANKLE",
]

# Generate all columns of the data frame

HEADERS = ["label"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


# In[11]:


def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg


def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width*2, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# In[12]:


VIDEO_PATH1 = "v2.mp4"



# In[13]:




# In[14]:


from tensorflow.keras.models import load_model
# Load model



# In[18]:


class SquatPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
    
    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].visibility
        ]
        
        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.hip = [landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].y]
        self.knee = [landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].y]
        self.ankle = [landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].y]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Squat Counter
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        squat_angle = int(calculate_angle(self.hip, self.knee, self.ankle))
        if squat_angle > self.stage_down_threshold:
            self.stage = "down"
        elif squat_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return (squat_angle)
shallow_squat_errors = []
knees_inward_errors = []
knees_forward_errors = []

# In[19]:



current_stage_sh = ""
current_stage_ki = ""
current_stage_kf = ""
prediction_probability_threshold = 0.6


VISIBILITY_THRESHOLD = 0.65


# Params for counter
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120
import pyautogui


# Init analysis class
left_knee_analysis =SquatPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

right_knee_analysis =SquatPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

inward_error_detected=False
forward_error_detected=False


def analyze_squat(video):

# Dump input scaler
    with open("E:/graduation/try flask/gym_eye/squat/input_scaler.pkl", "rb") as f2:
        sh_input_scaler = pickle.load(f2)
        
    with open("E:/graduation/try flask/gym_eye/squat/ki_input_scaler.pkl", "rb") as f2:
        ki_input_scaler = pickle.load(f2)
        
    with open("E:/graduation/try flask/gym_eye/squat/kf_input_scaler.pkl", "rb") as f2:
        kf_input_scaler = pickle.load(f2)

    shallow_model = load_model("E:/graduation/try flask/gym_eye/squat/shallow_squat_dp.h5")

    inward_model = load_model("E:/graduation/try flask/gym_eye/squat/ki_dp.h5")

    forward_model = load_model("E:/graduation/try flask/gym_eye/squat/kf_dp.h5")
    cap = cv2.VideoCapture(video)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break

            # Reduce size of a frame
            image = rescale_frame(image, 70)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                print("No human found")
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Make detection
            try:
                
                landmarks = results.pose_landmarks.landmark
                
                (left_squat_angle) = left_knee_analysis.analyze_pose(landmarks=landmarks, frame=image)
                (right_squat_angle) = right_knee_analysis.analyze_pose(landmarks=landmarks, frame=image)

                
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results)
                X = pd.DataFrame([row, ], columns=HEADERS[1:])
                X = pd.DataFrame(sh_input_scaler.transform(X))
                Y = pd.DataFrame([row, ], columns=HEADERS[1:])
                Y = pd.DataFrame(ki_input_scaler.transform(Y))
                Z = pd.DataFrame([row, ], columns=HEADERS[1:])
                Z = pd.DataFrame(kf_input_scaler.transform(Y))
                

                # Make prediction and its probability
                prediction_sh = shallow_model.predict(X)
                predicted_class_sh = np.argmax(prediction_sh, axis=1)[0]

                prediction_probability_sh = max(prediction_sh.tolist()[0])
                
                
                prediction_ki = inward_model.predict(Y)
                predicted_class_ki = np.argmax(prediction_ki, axis=1)[0]

                prediction_probability_ki = max(prediction_ki.tolist()[0])
                
                
                prediction_kf = forward_model.predict(Z)
                predicted_class_kf = np.argmax(prediction_kf, axis=1)[0]

                prediction_probability_kf = max(prediction_kf.tolist()[0])
                
                
                # Evaluate model prediction
                if predicted_class_sh == 0 and prediction_probability_sh >= prediction_probability_threshold:
                    current_stage_sh = "shallow squat"
                elif predicted_class_sh == 1 and prediction_probability_sh >= prediction_probability_threshold: 
                    current_stage_sh = "deep squat"
                else:
                    current_stage_sh = "UNK"
                    
                if predicted_class_ki == 0 and prediction_probability_ki >= prediction_probability_threshold:
                    current_stage_ki = "no inward error"
                    knees_inward_errors.append(1)
                elif predicted_class_ki == 1 and prediction_probability_ki >= prediction_probability_threshold: 
                    current_stage_ki = "knees inward"
                    knees_inward_errors.append(0)
                    if not inward_error_detected:
                    
                        cv2.imwrite("knees_inward_error.png", image)
                        inward_error_detected = True         
                else:
                    current_stage_ki = "UNK"
                    
                if predicted_class_kf == 0 and prediction_probability_kf >= prediction_probability_threshold:
                    current_stage_kf = "no forward error"
                    knees_forward_errors.append(1)
                    
                elif predicted_class_kf == 1 and prediction_probability_kf >= prediction_probability_threshold: 
                    current_stage_kf = "knees forward"
                    knees_forward_errors.append(0)
                    if not forward_error_detected:
                    
                        cv2.imwrite("knees_forward_error.png",image)
                        forward_error_detected = True                
                else:
                    current_stage_kf = "UNK"
        
            
                    
            except Exception as e:
                print(f"Error: {e}")
            
            cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
    for i in range (1, 5):
        cv2.waitKey(1)
    num_zeros = knees_inward_errors.count(0)
    num_ones = knees_inward_errors.count(1)
    ratio = num_zeros / (num_ones+num_zeros)
    knees_inward_error_ratio_percent = ratio
    
    num_zeros = knees_forward_errors.count(0)
    num_ones = knees_forward_errors.count(1)
    ratio = num_zeros / (num_ones+num_zeros)
    knees_forward_errors_ratio_percent= ratio
    return knees_inward_error_ratio_percent*100,knees_forward_errors_ratio_percent*100,left_knee_analysis.counter,right_knee_analysis.counter

# # Print all errors detected for each stage
# print("Shallow squat errors:", shallow_squat_errors)
# print("Knees inward errors:", knees_inward_errors)
# print("Knees forward errors:", knees_forward_errors)
# num_zeros = knees_inward_errors.count(0)
# num_ones = knees_inward_errors.count(1)
# ratio = num_zeros / (num_ones+num_zeros)
# ratio_percent = ratio
# print("Ratio of 0's to 1's: {:.2f}%".format(ratio_percent*100))

# num_zeros = knees_forward_errors.count(0)
# num_ones = knees_forward_errors.count(1)
# ratio = num_zeros / (num_ones+num_zeros)
# ratio_percent2 = ratio
# print("Ratio of 0's to 1's: {:.2f}%".format(ratio_percent*100))

# import json

# # Store the data in a dictionary
# data = {
#     "Ratio of 0's to 1's for knees inward errors": ratio_percent*100,
#     "Ratio of 0's to 1's for knees forward errors": ratio_percent2*100,
#     "Left knee count": left_knee_analysis.counter,
#     "Right knee count": right_knee_analysis.counter,
#     "forward_image":"imageURL",
#      "inward_image":"imageURL",
# }

# # Convert the dictionary to JSON format
# json_data = json.dumps(data)

# # Print the JSON data
# print(json_data)

