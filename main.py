import fastapi
import torch
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import os

from clock_model_color import ColorClockVAEHandler
from clock_model_mono import MonoClockVAEHandler

app = fastapi.FastAPI()

color_model_path = "clock-vae-color-140x-v1-500epoch.pth" # os.getenv("COLOR_MODEL_PATH")
mono_model_path = "clock-vae-mono-100x-v1-1000epoch.pth" # os.getenv("MONO_MODEL_PATH")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

color_model = ColorClockVAEHandler(model_path=color_model_path, device=device)
mono_model = MonoClockVAEHandler(model_path="clock-vae-mono-100x-v1-1000epoch.pth", size=100, device=device)

class TimeInput(BaseModel):
    hour: int
    minute: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/clock_captcha_color/")
async def clock_captcha_color(input_time: TimeInput):
    correct_image = color_model.generate_image(input_time.hour, input_time.minute)
    
    # Convert image to PNG format in a BytesIO buffer
    _, encoded_image = cv2.imencode(".png", correct_image)
    buffer = BytesIO(encoded_image.tobytes())
    buffer.seek(0)
    
    # Return a StreamingResponse with the PNG image
    return StreamingResponse(buffer, media_type="image/png")

@app.post("/clock_captcha_mono/")
async def clock_captcha_mono(input_time: TimeInput):
    correct_image = mono_model.generate_image(input_time.hour, input_time.minute)
    
    _, encoded_image = cv2.imencode(".png", correct_image)
    buffer = BytesIO(encoded_image.tobytes())
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")