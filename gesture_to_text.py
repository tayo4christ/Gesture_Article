import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Real-time gesture-to-text translator.")
    parser.add_argument("--model", default="data/gesture_model.pkl", help="Path to trained model (.pkl)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            label = "..."

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    X = np.array(coords).reshape(1, -1)
                    label = str(clf.predict(X)[0])

            cv2.putText(frame, f"Recognized: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Gesture-to-Text - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
