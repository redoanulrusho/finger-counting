import cv2
import numpy as np
import math
import streamlit as st

st.title("🖐️ Real-Time Finger Counting App")

# Checkbox to start/stop webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# HSV Range sliders (replacing cv2 trackbars)
st.sidebar.header("🎨 Color Adjustments (HSV Range)")
lower_h = st.sidebar.slider("Lower H", 0, 255, 0)
lower_s = st.sidebar.slider("Lower S", 0, 255, 0)
lower_v = st.sidebar.slider("Lower V", 0, 255, 0)
upper_h = st.sidebar.slider("Upper H", 0, 255, 255)
upper_s = st.sidebar.slider("Upper S", 0, 255, 255)
upper_v = st.sidebar.slider("Upper V", 0, 255, 255)

# Initialize webcam
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)  # mirror for natural view
    frame = cv2.resize(frame, (400, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtr = cv2.bitwise_and(frame, frame, mask=mask)

    mask1 = cv2.bitwise_not(mask)
    _, thresh = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=6)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                        if angle <= math.pi / 2:
                            finger_count += 1
                            cv2.circle(frame, far, 5, (0, 0, 255), -1)

    total_fingers = finger_count + 1 if finger_count > 0 else 0
    cv2.putText(frame, f"Fingers: {total_fingers}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert BGR → RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
