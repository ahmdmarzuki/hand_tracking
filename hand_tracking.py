import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(-1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    status, frame = cam.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # flippedFrame = (frame, 1)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)


    cv2.imshow("frame", frame)
    cv2.waitKey(1)

