#!/usr/bin/env python
# coding: utf-8

# In[22]:


import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[23]:


# Determine important landmarks for barbell row
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


# In[35]:


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

import cv2

def rescale_frame(frame, percent=1.0):
    '''
    Rescale a frame to a certain percentage compared to its original frame
    '''
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)

# In[36]:


VIDEO_PATH1 = "v3.mp4"
#VIDEO_PATH2 = "C:/Users/Alrowad/Exercise-correction/barbellrow_model/dataset/val/video_2023-05-07_17-51-28.mp4"


# In[37]:


# Load input scaler



# In[38]:


from tensorflow.keras.models import load_model
# Load model



# In[41]:


class BarbellPoseAnalysis:
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
        joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
        self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
        self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Barbellrow Counter
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        barbell_row_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if barbell_row_angle > self.stage_down_threshold:
            self.stage = "down"
        elif barbell_row_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return (barbell_row_angle)


# In[43]:


current_stage_L = ""
current_stage_T = ""
prediction_probability_threshold = 0.6

VISIBILITY_THRESHOLD = 0.65


# Params for counter
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120


# Init analysis class
left_arm_analysis = BarbellPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

right_arm_analysis = BarbellPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)
lValues=[]
tValues=[]

t_detected=False
l_detected=False

def analyzeBarbellRow(video):
    lumbar_model = load_model("E:/graduation/try flask/gym_eye/barbell_row/barbell_lumbar_dp.h5")
    torso_model = load_model("E:/graduation/try flask/gym_eye/barbell_row/barbell_torso_dp.h5")
    with open("E:/graduation/try flask/gym_eye/barbell_row/input_scaler_lumbar.pkl", "rb") as f:
        input_scaler_lumbar = pickle.load(f)

# Load input scaler
    with open("E:/graduation/try flask/gym_eye/barbell_row/input_scaler_torso.pkl", "rb") as f:
        input_scaler_torso = pickle.load(f)
    cap = cv2.VideoCapture(video)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Reduce size of a frame
            image = rescale_frame(image, 1)

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
                
                (left_barbell_row_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                (right_barbell_row_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results)
                X = pd.DataFrame([row, ], columns=HEADERS[1:])
                Y = pd.DataFrame([row, ], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler_lumbar.transform(X))
                Y = pd.DataFrame(input_scaler_torso.transform(Y))
                

                # Make prediction and its probability
                prediction_L = lumbar_model.predict(X)
                predicted_class_L = np.argmax(prediction_L, axis=1)[0]

                prediction_probability_L = max(prediction_L.tolist()[0])
                
                
                prediction_T = torso_model.predict(Y)
                predicted_class_T = np.argmax(prediction_T, axis=1)[0]

                prediction_probability_T = max(prediction_T.tolist()[0])
                


                # Evaluate model prediction
                # Evaluate model prediction
                if predicted_class_L == 0 and prediction_probability_L >= prediction_probability_threshold:
                    current_stage_L = "LC"
                    lValues.append(0)

                elif predicted_class_L == 1 and prediction_probability_L >= prediction_probability_threshold: 
                    current_stage_L = "LE"
                    lValues.append(1)
                    if not l_detected:
                        cv2.imwrite("L_error.png",image)
                        l_detected=True                
                else:
                    current_stage_L = "UNK"
                    
                if predicted_class_T == 0 and prediction_probability_T >= prediction_probability_threshold:
                    current_stage_T = "TC"
                    tValues.append(0)
                elif predicted_class_T == 1 and prediction_probability_T >= prediction_probability_threshold: 
                    current_stage_T = "TE"
                    tValues.append(1)
                    if not t_detected:
                        cv2.imwrite("T_error.png",image)
                        t_detected=True                 
                else:
                    current_stage_T = "UNK"
                        
            except Exception as e:
                print(f"Error: {e}")
            
            
            
    cap.release()
    cv2.destroyAllWindows()

    for i in range (1, 5):
        cv2.waitKey(1)
    left_counter = left_arm_analysis.counter
    right_counter = right_arm_analysis.counter
    l_num_zeros = lValues.count(0)
    l_num_ones = lValues.count(1)
    l_ratio = l_num_zeros / (l_num_ones + l_num_zeros)
    l_ratio_percent = l_ratio * 100
    t_num_zeros = tValues.count(0)
    t_num_ones = tValues.count(1)
    t_ratio = t_num_zeros / (t_num_ones + t_num_zeros)
    t_ratio_percent = t_ratio * 100
    return left_counter,right_counter,l_ratio_percent,t_ratio_percent





# import json

# # Define the variables

# print(lValues.count(0))
# print(lValues.count(1))
# # Create a Python dictionary with the values
# data = {
#     "left_counter": left_counter,
#     "right_counter": right_counter,
#     "l_ratio_percent": l_ratio_percent,
#     "t_ratio_percent": t_ratio_percent
# }

# # Convert the dictionary to a JSON object
# json_data = json.dumps(data)

# # Print the JSON object
# print(json_data)




