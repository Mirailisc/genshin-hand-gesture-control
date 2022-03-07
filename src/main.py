import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from tensorflow.python.keras.models import load_model

#OpenCV
webcam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
image = webcam.read()

#meidapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#tensorflow
model = load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

while webcam.isOpened():
    ret, frame = webcam.read()
    x , y, c = frame.shape 
    #Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Detect hands
    results = hands.process(image)
    
    #Render results
    if results.multi_hand_landmarks:
        landmarks = []
        for handslms in results.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
                                   
        #Prediction hand geature
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]
        
        print(className)

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        #Mouse and Keyboard Control
        if className == "okay":
            pyautogui.press('e')
            
        elif className == "peace":
            pyautogui.press('q')
            
        elif className == "fist":
            pyautogui.click(button='left')
        
    cv2.imshow("Hand Tranking", frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

webcam.release()
cv2.destroyAllWindows()