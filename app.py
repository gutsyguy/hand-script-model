import matplotlib.pyplot as plt
import numpy as np
import os 
import mediapipe as mp
import time
import cv2


mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing Utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Color Conversion BGR 2 RBG
    image.flags.writeable = False #Image is no longer writable
    results = model.process(image) #Make prediction
    image.flags.writeable = True #Image is now writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color Conversion RBG 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


cap = cv2.VideoCapture(0)
#Access mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read Feed
        ret, frame = cap.read()

        #make detections
        image, results = mediapipe_detection(frame, holistic) 


        #Draw landmarks
        draw_landmarks(image, results)

        # Show to screen 
        cv2.imshow("OPENCV Feed", image)

        #Break feed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()


draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))