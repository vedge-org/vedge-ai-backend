# face.py
import cv2
import dlib
import numpy as np

# face.py

class FaceRecognizer:
    def __init__(self):
        # dlib 모델 로드
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 이미지에서 얼굴 임베딩을 추출하는 함수
    def get_face_embedding_from_image(self, image_array: np.ndarray):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        face = faces[0]
        shape = self.sp(gray, face)
        face_descriptor = self.facerec.compute_face_descriptor(image_array, shape)
        return np.array(face_descriptor)

    # 두 임베딩 벡터의 유사도를 계산하는 함수
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.3):
        distance = np.linalg.norm(embedding1 - embedding2)
        is_similar = bool(distance < threshold)  # numpy.bool -> Python bool 타입으로 변환
        return {
            "distance": distance,
            "is_match": is_similar
        }
