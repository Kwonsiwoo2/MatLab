import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def add_blush(frame, landmarks):
    left_cheek_index = 1
    right_cheek_index = 1

    left_cheek_landmark = landmarks[left_cheek_index]
    left_cheek_point = (int(left_cheek_landmark.x * frame.shape[1]),
                        int(left_cheek_landmark.y * frame.shape[0]))

    right_cheek_landmark = landmarks[right_cheek_index]
    right_cheek_point = (int(right_cheek_landmark.x * frame.shape[1]),
                         int(right_cheek_landmark.y * frame.shape[0]))

    for point in [left_cheek_point, right_cheek_point]:
        overlay = np.zeros_like(frame)
        alpha = 0.5
        radius = 15
        color = (0, 0, 255)

        cv2.circle(overlay, point, radius, color, -1)
        blurred_overlay = cv2.GaussianBlur(overlay, (91, 91), 0)
        cv2.addWeighted(blurred_overlay, alpha, frame, 1, 0, frame)

def get_eye_center(landmarks, indices, frame):
    x_coords = [int(landmarks[i].x * frame.shape[1]) for i in indices]
    y_coords = [int(landmarks[i].y * frame.shape[0]) for i in indices]
    return int(np.mean(x_coords)), int(np.mean(y_coords))

def calculate_angle(landmarks, frame):
    left_eye = get_eye_center(landmarks, [33, 133], frame)
    right_eye = get_eye_center(landmarks, [362, 263], frame)
    
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_image(image, angle):
    # 이미지의 중심을 기준으로 회전 행렬 생성
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 행렬을 사용하여 이미지 회전
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return rotated

def overlay_sunglasses(frame, landmarks):
    sunglasses_img = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)
    sunglasses_img = cv2.flip(sunglasses_img, 1)
    h, w, _ = frame.shape
    
    left_eye_indices = [33, 133]
    right_eye_indices = [362, 263]

    left_eye_x1 = int(landmarks[left_eye_indices[0]].x * w)
    left_eye_y1 = int(landmarks[left_eye_indices[0]].y * h)
    left_eye_x2 = int(landmarks[left_eye_indices[1]].x * w)
    left_eye_y2 = int(landmarks[left_eye_indices[1]].y * h)
    
    right_eye_x1 = int(landmarks[right_eye_indices[0]].x * w)
    right_eye_y1 = int(landmarks[right_eye_indices[0]].y * h)
    right_eye_x2 = int(landmarks[right_eye_indices[1]].x * w)
    right_eye_y2 = int(landmarks[right_eye_indices[1]].y * h)

    x1 = min(left_eye_x1, right_eye_x1)
    y1 = min(left_eye_y1, right_eye_y1)
    x2 = max(left_eye_x2, right_eye_x2)
    y2 = max(left_eye_y2, right_eye_y2)
    
    glasses_width = int((x2 - x1) * 2.3)  # 크기를 2배로 확대
    glasses_height = int(0.8 * glasses_width)

    if glasses_width > 0 and glasses_height > 0:
        overlay = cv2.resize(sunglasses_img, (glasses_width, glasses_height))
    else:
        return

    x_center = (x1 + x2 + 16) // 2
    y_center = (y1 + y2+10) // 2
    x1 = x_center - glasses_width // 2
    y1 = y_center - glasses_height // 2
    x2 = x1 + glasses_width
    y2 = y1 + glasses_height
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return
    
    angle = calculate_angle(landmarks, frame)
    overlay_rotated = rotate_image(overlay, -angle)  # 각도를 반전시킵니다.
    
    overlay_img = frame[y1:y2, x1:x2]
    if overlay_img.shape[0] > 0 and overlay_img.shape[1] > 0:
        alpha_s = overlay_rotated[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            overlay_img[:, :, c] = (alpha_s * overlay_rotated[:, :, c] + alpha_l * overlay_img[:, :, c])

        frame[y1:y2, x1:x2] = overlay_img

def overlay_rabbit_ears(frame, landmarks):
    rabbit_ear_img = cv2.imread("rabbit_ear2.png", cv2.IMREAD_UNCHANGED)
    if rabbit_ear_img is None:
        raise FileNotFoundError("rabbit_ear2.png 파일을 찾을 수 없습니다.")
    
    forehead_index = 10  # 이마 위치의 랜드마크 인덱스
    forehead_landmark = landmarks[forehead_index]
    
    h, w, _ = frame.shape
    forehead_x = int(forehead_landmark.x * w)
    forehead_y = int(forehead_landmark.y * h)

    ear_width, ear_height = rabbit_ear_img.shape[1], rabbit_ear_img.shape[0]
    x1, y2 = forehead_x - ear_width // 2, forehead_y
    x2, y1 = x1 + ear_width, y2 - ear_height

    # 화면 경계를 넘지 않도록 조정
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped_width = x2 - x1
    cropped_height = y2 - y1

    if cropped_width > 0 and cropped_height > 0:
        rabbit_ear_cropped = rabbit_ear_img[-cropped_height:, :cropped_width]
        alpha_s = rabbit_ear_cropped[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * rabbit_ear_cropped[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

def apply_background_change(frame, landmarks, background_image):
    face_landmarks_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
        93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array([(int(landmarks[idx].x * frame.shape[1]), int(landmarks[idx].y * frame.shape[0])) for idx in face_landmarks_indices], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], (255, 255, 255))

    resized_background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    inverse_mask = cv2.bitwise_not(mask)

    face_area = cv2.bitwise_and(frame, frame, mask=mask)
    background_area = cv2.bitwise_and(resized_background, resized_background, mask=inverse_mask)

    output = cv2.add(face_area, background_area)

    return output