import cv2
import mediapipe as mp

# MediaPipe 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)

# 창 이름 설정
window_name = 'Face Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 480, 480)

# MediaPipe 얼굴 메쉬 모델 초기화
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 얼굴 메쉬 추론 수행
        results = face_mesh.process(image)

        # 이미지를 다시 BGR로 변환하여 그리기
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 얼굴 랜드마크 그리기
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 랜드마크 그리기 (빨강색), 연결선 그리기 (초록색)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255)),  # 빨강색
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))  # 초록색
                )

        # 프레임 크기 조절 (원하는 크기로 설정)
        resized_frame = cv2.resize(image, (480, 480))

        # 결과 이미지 표시
        cv2.imshow(window_name, resized_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 웹캠 릴리스 및 창 닫기
cap.release()
cv2.destroyAllWindows()
