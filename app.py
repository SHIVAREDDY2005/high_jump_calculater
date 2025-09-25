# import cv2
# import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# cap = cv2.VideoCapture(0)

# ground_level = None
# max_jump = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb)

#     if results.pose_landmarks:
#         h, w, _ = frame.shape
#         lm = results.pose_landmarks.landmark

#         # Get ankle positions
#         left_ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
#         right_ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h
#         ankle_y = (left_ankle_y + right_ankle_y) / 2

#         # Set baseline ground level (when standing)
#         if ground_level is None:
#             ground_level = ankle_y

#         # Calculate jump height in pixels
#         jump_height_px = ground_level - ankle_y

#         # Update maximum jump detected
#         if jump_height_px > max_jump:
#             max_jump = jump_height_px

#         # Display results
#         cv2.putText(frame, f"Jump: {int(jump_height_px)} px",
#                     (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#         cv2.putText(frame, f"Max Jump: {int(max_jump)} px",
#                     (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#     cv2.imshow("High Jump Detection", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import tempfile
import streamlit as st
import numpy as np

st.title("🏃 High Jump Detection by khelsuthra (Hip-based)")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Initialize mediapipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    ground_level = None
    max_jump = 0

    # To display video frames in Streamlit
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark

            # ✅ Use hips instead of ankles
            left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
            right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
            hip_y = (left_hip_y + right_hip_y) / 2

            # Set baseline ground level
            if ground_level is None:
                ground_level = hip_y

            # Calculate jump height in pixels
            jump_height_px = ground_level - hip_y

            if jump_height_px > max_jump:
                max_jump = jump_height_px

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Show jump info on video
            cv2.putText(frame, f"Jump: {int(jump_height_px)} px",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Max Jump: {int(max_jump)} px",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Convert BGR → RGB for Streamlit display
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    st.success(f"✅ Maximum Jump Height Detected (Hip-based): {int(max_jump)} pixels")
