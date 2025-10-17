import streamlit as st
import cv2
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Page setup
st.set_page_config(page_title="🖐️ Finger Counting", layout="centered")
st.title("🖐️ Live Finger Counting via Camera")

# HSV sliders in sidebar
st.sidebar.header("🎨 HSV Range")
lower_h = st.sidebar.slider("Lower H", 0, 255, 0)
lower_s = st.sidebar.slider("Lower S", 0, 255, 30)
lower_v = st.sidebar.slider("Lower V", 0, 255, 60)
upper_h = st.sidebar.slider("Upper H", 0, 255, 20)
upper_s = st.sidebar.slider("Upper S", 0, 255, 255)
upper_v = st.sidebar.slider("Upper V", 0, 255, 255)

# STUN server config for WebRTC
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Finger counting logic
class FingerCounter(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (400, 400))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        mask_inv = cv2.bitwise_not(mask)
        _, thresh = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=6)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        finger_count = 0
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000:
                hull = cv2.convexHull(max_contour, returnPoints=False)
                if hull is not None and len(hull) > 3:
                    defects = cv2.convexityDefects(max_contour, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(max_contour[s][0])
                            end = tuple(max_contour[e][0])
                            far = tuple(max_contour[f][0])
                            a = math.dist(start, end)
                            b = math.dist(start, far)
                            c = math.dist(end, far)
                            if b != 0 and c != 0:
                                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
                                if angle <= math.pi / 2:
                                    finger_count += 1
                                    cv2.circle(img, far, 5, (0, 0, 255), -1)

        total_fingers = finger_count + 1 if finger_count > 0 else 0
        cv2.putText(img, f"Fingers: {total_fingers}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        return img

# Start webcam stream
webrtc_streamer(
    key="finger-counter",
    video_transformer_factory=FingerCounter,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
