import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        flippedFrame = cv2.flip(frame, 1)
        imgRGB = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(flippedFrame, landmarks, self.mpHands.HAND_CONNECTIONS)

        return flippedFrame
    
    def findPosition(self, frame, handNo=0, draw=True):

        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, channel = frame.shape
                # koordinat dari lm kisaran 0-1, jadi harus diubah sesuai dengan height dan width screen
                cx, cy = int(lm.x*width), int(lm.y*height)
                landmarkList.append([id, cx, cy])
                
                # if id == 0:
                # if draw:
                #    cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return landmarkList


def main():
    pTime = 0
    cTime = 0

    cam = cv2.VideoCapture(0)  # Gunakan 0 untuk kamera utama

    if not cam.isOpened():
        print("Error: Camera not found!")
        exit()

    detector = HandDetector()

    while True:
        status, frame = cam.read()
        if not status:
            print("Error: Failed to capture frame")
            continue  # Skip loop jika gagal menangkap frame

        frame = detector.findHands(frame)
        landmarkList = detector.findPosition(frame)

        if len(landmarkList) != 0:
           print(landmarkList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
