import cv2
import mediapipe as mp
import tempfile
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# ---------------- Processing Functions ----------------
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

                # Extract hips
                h, w, _ = image.shape
                lm = results.pose_landmarks.landmark
                left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
                hip_y = (left_hip_y + right_hip_y) / 2

                # Ground baseline
                if ground_level is None:
                    ground_level = hip_y

                # Jump height in pixels
                jump_height_px = ground_level - hip_y
                if jump_height_px > max_jump:
                    max_jump = jump_height_px

                cv2.putText(image, f'Jump: {int(jump_height_px)} px', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(image, f'Max Jump: {int(max_jump)} px', (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
    return max_jump


# ---------------- Live Video Processor ----------------
class JumpDetector(VideoProcessorBase):
    def __init__(self):
        self.ground_level = None
        self.max_jump = 0
        self.pose = mp_pose.Pose(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = image.shape
            lm = results.pose_landmarks.landmark
            left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
            right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
            hip_y = (left_hip_y + right_hip_y) / 2

            if self.ground_level is None:
                self.ground_level = hip_y

            jump_height_px = self.ground_level - hip_y
            if jump_height_px > self.max_jump:
                self.max_jump = jump_height_px

            cv2.putText(image, f'Jump: {int(jump_height_px)} px', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(image, f'Max Jump: {int(self.max_jump)} px', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ---------------- Streamlit UI ----------------
def main():
    st.title("🏃 High Jump Detection (Hip-based)")

    mode = st.radio("Choose mode:", ["📹 Upload Video", "🎥 Live Camera"])

    if mode == "📹 Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.info("Processing video... ⏳")
            placeholder = st.empty()
            max_jump = process_video(tfile.name, placeholder)
            st.success(f"✅ Done! Maximum Jump Height Detected: {int(max_jump)} px")

    elif mode == "🎥 Live Camera":
        st.info("Live mode started... jump in front of your webcam 🎥")
        webrtc_streamer(
            key="jump-detector",
            video_processor_factory=JumpDetector,
            media_stream_constraints={"video": True, "audio": False},
        )


if __name__ == "__main__":
    main()
