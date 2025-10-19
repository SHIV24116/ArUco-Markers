import cv2

# ---- [ SETTINGS ] ----
# Use your IP Webcam stream URL
camera_url = "http://<Wifi IP>:<Port>/video"  #Use 'camera_url = 0' for using ondevice camera

# ArUco dictionary
aruco_dict_type = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

# ---- [ PARAMETERS TUNING ] ----
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 53
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.02
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05
parameters.minDistanceToBorder = 3
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# ---- [ OPEN CAMERA STREAM ] ----
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("❌ Could not open mobile camera stream.")
    print("➡️ Make sure:")
    print("   1. Your phone & PC are on same Wi-Fi.")
    print("   2. The IP Webcam app is running.")
    print("   3. The URL is correct (try in a browser).")
    exit()

print("✅ Mobile camera connected.")
print("Press 'q' to quit.\n")

# ---- [ MAIN LOOP ] ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to receive frame.")
        break

    frame = cv2.resize(frame, (960, 540))

    # Convert to grayscale + preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers + coordinates
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            center = pts.mean(axis=0).astype(int)
            cx, cy = int(center[0]), int(center[1])

            # Draw bounding box and center point
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Display ID and coordinates
            text = f"(x:{cx}, y:{cy})"
            cv2.putText(frame, text, (cx-10, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Print all marker info to terminal
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            center = pts.mean(axis=0).astype(int)
            print(f"✅ ID {ids[i][0]} -> Center: ({center[0]:.1f}, {center[1]:.1f})")

    else:
        print("No markers detected.")

    # Show live feed
    cv2.imshow("📷 ArUco Detection Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- [ CLEANUP ] ----
cap.release()
cv2.destroyAllWindows()

