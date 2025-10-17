import cv2
import numpy as np
import math
import streamlit as st
from PIL import Image

st.title("Finger Counting with HSV Color Space")

# Sidebar sliders for HSV and threshold values
st.sidebar.header("HSV Color Thresholds")

l_h = st.sidebar.slider("Lower Hue", 0, 255, 0)
l_s = st.sidebar.slider("Lower Saturation", 0, 255, 48)
l_v = st.sidebar.slider("Lower Value", 0, 255, 80)

u_h = st.sidebar.slider("Upper Hue", 0, 255, 20)
u_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
u_v = st.sidebar.slider("Upper Value", 0, 255, 255)

thresh_val = st.sidebar.slider("Threshold", 0, 255, 127)

# Camera input widget (works on mobile and desktop browsers)
img_file_buffer = st.camera_input("Take a picture or use camera")

if img_file_buffer is not None:
    # Convert the uploaded image to OpenCV format
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (400, 400))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtr = cv2.bitwise_and(frame, frame, mask=mask)

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

    # Convert to RGB for displaying in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=finger_count_text)
else:
    st.write("Please take a picture or enable your camera.")
