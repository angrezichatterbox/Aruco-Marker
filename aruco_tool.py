import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import sys
import os

def generate_marker(marker_id, size=200, output_path="marker.png", dict_type=aruco.DICT_6X6_250):
    """
    Generate an ArUco marker and save it to a file.
    """
    # Load the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    
    # Generate the marker
    marker_image = np.zeros((size, size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add a white border around the marker (padding) so it can be detected correctly
    border_size = int(size * 0.1) # 10% border
    marker_image = cv2.copyMakeBorder(marker_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Save the marker
    cv2.imwrite(output_path, marker_image)
    print(f"Marker ID {marker_id} generated and saved to {output_path}")
import numpy as np
import cv2

def draw_cube(image, rvec, tvec, camera_matrix, dist_coeffs, marker_length, z_offset=0.015):
    """
    Draw a solid 3D cube with a Z-axis offset.
    Uses painter's algorithm to sort faces by depth (farthest drawn first).
    """
    half_l = marker_length / 2.0

    axis = np.float32([
        [-half_l,  half_l, z_offset],
        [ half_l,  half_l, z_offset],
        [ half_l, -half_l, z_offset],
        [-half_l, -half_l, z_offset],
        [-half_l,  half_l, z_offset + marker_length],
        [ half_l,  half_l, z_offset + marker_length],
        [ half_l, -half_l, z_offset + marker_length],
        [-half_l, -half_l, z_offset + marker_length]
    ])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Rotate the 3D points into camera space to get real depth
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ axis.T).T + tvec.flatten()  # shape (8, 3)

    face_indices = [
        [0, 1, 2, 3],  # Bottom
        [0, 1, 5, 4],  # Side 1
        [1, 2, 6, 5],  # Side 2
        [2, 3, 7, 6],  # Side 3
        [3, 0, 4, 7],  # Side 4
        [4, 5, 6, 7],  # Top
    ]

    colors = [
        (80,  80,  80),   # Bottom  — Shadow Gray
        (50,  50,  220),  # Side 1  — Blue
        (50,  220, 50),   # Side 2  — Green
        (20,  20,  150),  # Side 3  — Dark Blue
        (20,  150, 20),   # Side 4  — Dark Green
        (255, 150, 50),   # Top     — Orange
    ]

    # Compute each face's average Z depth in camera space
    face_depths = []
    for i, fi in enumerate(face_indices):
        avg_z = np.mean([pts_cam[j][2] for j in fi])
        face_depths.append((avg_z, i))

    # Sort farthest → nearest (painter's algorithm)
    face_depths.sort(key=lambda x: -x[0])

    # Draw faces back-to-front
    for avg_z, i in face_depths:
        fi = face_indices[i]
        face_array = np.array([imgpts[j] for j in fi], dtype=np.int32)
        cv2.fillPoly(image, [face_array], colors[i])

    # Draw outlines on top of all filled faces
    for i, fi in enumerate(face_indices):
        face_array = np.array([imgpts[j] for j in fi], dtype=np.int32)
        cv2.polylines(image, [face_array], True, (0, 0, 0), 2)

    return image

def process_ar(image, corners, marker_length=0.05):
    """
    Estimate pose and draw a 3D cube for each detected marker.
    Uses a default camera matrix based on image size.
    """
    h, w = image.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    half_l = marker_length / 2.0
    # Marker object points (top-left, top-right, bottom-right, bottom-left)
    obj_points = np.array([
        [-half_l,  half_l, 0],
        [ half_l,  half_l, 0],
        [ half_l, -half_l, 0],
        [-half_l, -half_l, 0]
    ], dtype=np.float32)

    for i in range(len(corners)):
        success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], camera_matrix, dist_coeffs)
        if success:
            draw_cube(image, rvec, tvec, camera_matrix, dist_coeffs, marker_length)

def detect_image(image_path, dict_type=aruco.DICT_6X6_250, headless=False, apply_ar=False):
    """
    Detect ArUco markers in a static image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        sys.exit(1)

    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    
    if ids is not None:
        print(f"Detected {len(ids)} marker(s). IDs: {ids.flatten()}")
        # Draw detected markers on the image
        aruco.drawDetectedMarkers(image, corners, ids)
        
        if apply_ar:
            process_ar(image, corners)
        
        output_file = "detected_" + os.path.basename(image_path)
        cv2.imwrite(output_file, image)
        print(f"Saved detection result to {output_file}")
        
        if not headless:
            # Display the result
            cv2.imshow("Detected ArUco Markers", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No ArUco markers detected in the image.")

def detect_webcam(dict_type=aruco.DICT_6X6_250, camera_index=0, apply_ar=False):
    """
    Detect ArUco markers using a webcam feed.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (index {camera_index}).")
        sys.exit(1)
        
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Starting webcam detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
        
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            if apply_ar:
                process_ar(frame, corners)
            
        cv2.imshow('ArUco Detection (Webcam)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="ArUco Marker Utility")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Subparser for generating a marker
    parser_gen = subparsers.add_parser("generate", help="Generate an ArUco marker")
    parser_gen.add_argument("--id", type=int, default=1, help="Marker ID (default: 1)")
    parser_gen.add_argument("--size", type=int, default=200, help="Marker size in pixels (default: 200)")
    parser_gen.add_argument("--out", type=str, default="marker.png", help="Output file path (default: marker.png)")
    
    # Subparser for detecting from an image
    parser_det_img = subparsers.add_parser("detect-img", help="Detect ArUco markers in an image")
    parser_det_img.add_argument("image", type=str, help="Path to the image file")
    parser_det_img.add_argument("--headless", action="store_true", help="Do not display the image window")
    parser_det_img.add_argument("--ar", action="store_true", help="Overlay a 3D cube (Augmented Reality)")
    
    # Subparser for detecting from a webcam
    parser_det_webcam = subparsers.add_parser("detect-webcam", help="Detect ArUco markers using the webcam")
    parser_det_webcam.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser_det_webcam.add_argument("--ar", action="store_true", help="Overlay a 3D cube (Augmented Reality)")

    args = parser.parse_args()

    # We are using DICT_6X6_250 for this script
    dict_type = aruco.DICT_6X6_250

    if args.command == "generate":
        generate_marker(args.id, args.size, args.out, dict_type)
    elif args.command == "detect-img":
        detect_image(args.image, dict_type, args.headless, args.ar)
    elif args.command == "detect-webcam":
        detect_webcam(dict_type, args.camera, args.ar)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
