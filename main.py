from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
import torch
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import dlib
import numpy as np
from typing import List

from clock_model_color import ColorClockVAEHandler
from clock_model_mono import MonoClockVAEHandler
from face_embedding import FaceRecognizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

color_model_path = "clock-vae-color-140x-v1-500epoch.pth" # os.getenv("COLOR_MODEL_PATH")
mono_model_path = "clock-vae-mono-100x-v1-1000epoch.pth" # os.getenv("MONO_MODEL_PATH")
color_model = ColorClockVAEHandler(model_path=color_model_path, device=device)
mono_model = MonoClockVAEHandler(model_path="clock-vae-mono-100x-v1-1000epoch.pth", size=100, device=device)

face_recognizer = FaceRecognizer()

app = FastAPI()

class TimeRequst(BaseModel):
    hour: int
    minute: int

class SimilarityRequest(BaseModel):
    embedding1: List[float]
    embedding2: List[float]


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/clock_captcha_color/")
async def clock_captcha_color(input_time: TimeRequst):
    correct_image = color_model.generate_image(input_time.hour, input_time.minute)
    
    # Convert image to PNG format in a BytesIO buffer
    _, encoded_image = cv2.imencode(".png", correct_image)
    buffer = BytesIO(encoded_image.tobytes())
    buffer.seek(0)
    
    # Return a StreamingResponse with the PNG image
    return StreamingResponse(buffer, media_type="image/png")

@app.post("/clock_captcha_mono/")
async def clock_captcha_mono(input_time: TimeRequst):
    correct_image = mono_model.generate_image(input_time.hour, input_time.minute)
    
    _, encoded_image = cv2.imencode(".png", correct_image)
    buffer = BytesIO(encoded_image.tobytes())
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")

@app.post("/extract_embedding/")
async def extract_embedding(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    embedding = face_recognizer.get_face_embedding_from_image(image)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="이미지에서 얼굴을 찾을 수 없습니다.")

    return {"embedding": embedding.tolist()}

@app.post("/check_similarity/")
async def check_similarity(request: SimilarityRequest):
    embedding1 = np.array(request.embedding1)
    embedding2 = np.array(request.embedding2)

    result = face_recognizer.calculate_similarity(embedding1, embedding2)
    return result

@app.post("/check_two_face/")
async def check_two_face(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    np_img1 = np.frombuffer(contents1, np.uint8)
    np_img2 = np.frombuffer(contents2, np.uint8)

    image1 = cv2.imdecode(np_img1, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np_img2, cv2.IMREAD_COLOR)

    embedding1 = face_recognizer.get_face_embedding_from_image(image1)
    embedding2 = face_recognizer.get_face_embedding_from_image(image2)

    if embedding1 is None or embedding2 is None:
        raise HTTPException(status_code=400, detail="이미지에서 얼굴을 찾을 수 없습니다.")

    result = face_recognizer.calculate_similarity(embedding1, embedding2)
    return result