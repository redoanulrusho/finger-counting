import cv2
import numpy as np
import math
import streamlit as st

st.title("Live Finger Counting with HSV Color Space")

# Sidebar for HSV and threshold values
l_h = st.sidebar.slider("Lower Hue", 0, 255, 0)
l_s = st.sidebar.slider("Lower Saturation", 0, 255, 48)
l_v = st.sidebar.slider("Lower Value", 0, 255, 80)

u_h = st.sidebar.slider("Upper Hue", 0, 255, 20)
u_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
u_v = st.sidebar.slider("Upper Value", 0, 255, 255)

thresh_val = st.sidebar.slider("Threshold", 0, 255, 127)

# Camera toggle switch
camera_on = st.checkbox("Turn Camera ON/OFF")

# Placeholder for video frames
frame_placeholder = st.empty()

if camera_on:
    cap = cv2.VideoCapture(0)  # Open default webcam

    if not cap.isOpened():
        st.error("Unable to open webcam")
    else:
        while camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            frame = cv2.resize(frame, (400, 400))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_bound = np.array([l_h, l_s, l_v])
            upper_bound = np.array([u_h, u_s, u_v])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask1 = cv2.bitwise_not(mask)
            ret, thresh = cv2.threshold(mask1, thresh_val, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=6)

            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            finger_count_text = "Fingers: 0"
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 1000:
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                    hull = cv2.convexHull(max_contour, returnPoints=False)
                    if hull is not None and len(hull) > 3:
                        defects = cv2.convexityDefects(max_contour, hull)
                        finger_count = 0
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

                        total_fingers = finger_count + 1
                        finger_count_text = f"Fingers: {total_fingers}"
                        cv2.putText(frame, finger_count_text, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption=finger_count_text)

            # Update camera_on status from the checkbox
            camera_on = st.checkbox("Turn Camera ON/OFF", value=True)

        cap.release()
else:
    st.write("Camera is OFF. Toggle the switch to start.")
