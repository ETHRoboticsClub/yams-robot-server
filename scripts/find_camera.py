#!/usr/bin/env python3
"""
Camera Index Finder
Scans for available camera indices and shows a brief preview of each camera.
This helps identify which index corresponds to which physical camera.
"""

import time

import cv2


def find_cameras():
    """Find all available camera indices and display basic info."""
    available_cameras = []

    print("Scanning for available cameras...")
    print("=" * 50)

    # Check indices 0 through 9 (covers most systems)
    for index in range(10):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera {index}: Available")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")

            # Try to capture a frame to verify it's working
            ret, frame = cap.read()
            if ret:
                print(f"  Status: Working (frame captured successfully)")
                available_cameras.append(index)
            else:
                print(f"  Status: Available but no frame captured")

            cap.release()
            print()
        else:
            # Uncomment the line below if you want to see unavailable indices
            # print(f"Camera {index}: Not available")
            pass

    return available_cameras


def preview_cameras(camera_indices):
    """Show a brief preview of each available camera."""
    if not camera_indices:
        print("No cameras found!")
        return

    print(f"Found {len(camera_indices)} camera(s): {camera_indices}")
    print("\nStarting camera preview...")
    print("Each camera will be shown for 3 seconds.")
    print("Press any key to skip to the next camera or 'q' to quit early.")
    print("=" * 50)

    for index in camera_indices:
        print(f"\nShowing Camera {index}...")

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Failed to open camera {index}")
            continue

        # Show camera feed for a few seconds
        start_time = time.time()
        while time.time() - start_time < 3:  # Show for 3 seconds
            ret, frame = cap.read()
            if ret:
                # Add text overlay to identify the camera
                cv2.putText(
                    frame,
                    f"Camera {index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press any key to continue",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(f"Camera {index} Preview", frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Any key pressed
                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    break
            else:
                print(f"Failed to capture frame from camera {index}")
                break

        cap.release()
        cv2.destroyWindow(f"Camera {index} Preview")
        time.sleep(0.5)  # Brief pause between cameras


def main():
    print("Camera Index Finder")
    print("This tool will help you identify which camera indices are available")
    print("and which physical camera each index corresponds to.\n")

    # Find available cameras
    available_cameras = find_cameras()

    if not available_cameras:
        print("No working cameras found!")
        print("Make sure your cameras are connected and not being used by other applications.")
        return

    # Ask user if they want to see camera previews
    if len(available_cameras) > 1:
        response = input(
            f"Would you like to see previews of the {len(available_cameras)} cameras? (y/n): "
        ).lower()
        if response in ["y", "yes"]:
            preview_cameras(available_cameras)
    else:
        print("Only one camera found. You can use index 0 for your script.")

    print(f"\nSummary: Available camera indices are: {available_cameras}")
    print("Use these indices in your show_both_cameras.py script.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
