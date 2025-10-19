import cv2
import cv2.aruco as aruco

# Ask user for marker ID
marker_id = int(input("Enter the marker ID (0–49): "))
marker_size = 400  # pixels

# Load predefined dictionary (4x4_50 has 50 unique markers)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

marker_padded = cv2.copyMakeBorder(marker_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)

# Show the marker
cv2.imshow(f"Aruco Marker ID {marker_id}", marker_padded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the marker as an image
cv2.imwrite(f"marker_ID_{marker_id}.png", marker_padded)
print(f"Marker saved as marker_ID_{marker_id}.png")
