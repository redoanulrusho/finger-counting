import cv2
import numpy as np
import math
import streamlit as st

st.set_page_config(page_title="Finger Counting", layout="centered")

st.title("🖐️ Live Finger Counting via Camera (Android Supported)")

run = st.checkbox("▶️ Start Live Detection")
FRAME_WINDOW = st.image([])

# HSV sliders
st.sidebar.header("🎨 HSV Range")
l_h = st.sidebar.slider("Lower H", 0, 255, 0)
l_s = st.sidebar.slider("Lower S", 0, 255, 30)
l_v = st.sidebar.slider("Lower V", 0, 255, 60)
u_h = st.sidebar.slider("Upper H", 0, 255, 20)
u_s = st.sidebar.slider("Upper S", 0, 255, 255)
u_v = st.sidebar.slider("Upper V", 0, 255, 255)

# Access webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    st.error("❌ Cannot access webcam. Please enable camera permission.")
    st.stop()

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (400, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask1 = cv2.bitwise_not(mask)
    _, thresh = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=6)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    finger_count = 0
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
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
                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                            if angle <= math.pi / 2:
                                finger_count += 1
                                cv2.circle(frame, far, 5, (0, 0, 255), -1)

    total_fingers = finger_count + 1 if finger_count > 0 else 0
    cv2.putText(frame, f"Fingers: {total_fingers}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
