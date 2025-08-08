import argparse
import csv
import os
import time

import cv2
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser(description="Collect hand landmark data for a gesture label.")
    parser.add_argument("--label", required=True, help="Name of the gesture class (e.g., thumbs_up)")
    parser.add_argument("--samples", type=int, default=200, help="Number of frames to capture")
    parser.add_argument("--outfile", default="data/gesture_data.csv", help="CSV output path")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    headers = []
    for i in range(21):
        headers += [f"x{i}", f"y{i}", f"z{i}"]
    headers += ["label"]

    # Create file if it doesn't exist
    file_exists = os.path.isfile(args.outfile)
    if not file_exists:
        with open(args.outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    collected = 0
    cooldown = 0.3  # seconds between captures
    last_capture = 0.0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    now = time.time()
                    if now - last_capture >= cooldown and collected < args.samples:
                        row = []
                        for lm in hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                        row.append(args.label)
                        with open(args.outfile, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                        collected += 1
                        last_capture = now

            cv2.putText(frame, f"Label: {args.label}  Collected: {collected}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Collecting Data - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Collected {collected} samples for label '{args.label}'.")

if __name__ == "__main__":
    main()
