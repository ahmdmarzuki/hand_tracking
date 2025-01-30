import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(-1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    status, frame = cam.read()
    flippedFrame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # flippedFrame = (frame, 1)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                # print(id, lm)
                height, width, channel = flippedFrame.shape

                # koordinat dari lm kisaran 0-1, jadi harus diubah sesuai dengan height dan width screen
                cx, cy = int(lm.x*width), int(lm.y*height)

                # if id == 0:
                #     cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(flippedFrame, landmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(flippedFrame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("frame", flippedFrame)
    cv2.waitKey(1)

