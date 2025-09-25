import mediapipe as mp
import tempfile
import streamlit as st
import numpy as np
import imageio

st.title("🏃 High Jump Detection (Hip-based, Streamlit Only)")

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

    # Open video with imageio
    reader = imageio.get_reader(video_path)

    ground_level = None
    max_jump = 0

    # To display video frames in Streamlit
    stframe = st.empty()

    for frame in reader:
        # Convert frame to RGB numpy array
        rgb = np.array(frame)
        results = pose.process(rgb)

        if results.pose_landmarks:
            h, w, _ = rgb.shape
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

            # Draw landmarks on frame
            annotated = rgb.copy()
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Show frame in Streamlit
            stframe.image(annotated, channels="RGB")

    st.success(f"✅ Maximum Jump Height Detected (Hip-based): {int(max_jump)} pixels")
