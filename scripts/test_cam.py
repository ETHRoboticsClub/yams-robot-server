import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description="Display camera feed")
    parser.add_argument(
        "--index", "-i", type=int, default=0, help="Camera index (default: 0)"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.index}")
        return

    print(f"Camera {args.index} opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        cv2.imshow(f"Camera {args.index}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
