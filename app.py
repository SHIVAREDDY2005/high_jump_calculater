import cv2
import mediapipe as mp
import tempfile
import streamlit as st

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# ---------------- Processing Function ----------------
def process_video(video_path, placeholder):
    cap = cv2.VideoCapture(video_path)

    ground_level = None
    max_jump = 0

    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Extract hip positions
                h, w, _ = image.shape
                lm = results.pose_landmarks.landmark

                left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
                hip_y = (left_hip_y + right_hip_y) / 2

                # Set baseline ground level
                if ground_level is None:
                    ground_level = hip_y

                # Calculate jump height
                jump_height_px = ground_level - hip_y
                if jump_height_px > max_jump:
                    max_jump = jump_height_px

                # Show jump info
                cv2.putText(image, f"Jump: {int(jump_height_px)} px",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Max Jump: {int(max_jump)} px",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
    return max_jump


# ---------------- Streamlit UI ----------------
def main():
    st.title("🏃 High Jump Detection (Hip-based)")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.info("Processing video... ⏳")
        placeholder = st.empty()
        max_jump = process_video(tfile.name, placeholder)

        st.success(f"✅ Maximum Jump Height Detected: {int(max_jump)} pixels")


if __name__ == "__main__":
    main()
