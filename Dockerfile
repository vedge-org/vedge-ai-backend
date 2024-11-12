# 베이스 이미지
FROM python:3.11

# 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 최신 pip 버전으로 업그레이드
RUN pip install --upgrade pip

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt


# Hugging Face Hub에서 .pth 파일 다운로드
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='chaehoyu/clock-vae-color-140x-v1', filename='clock-vae-color-140x-v1-500epoch.pth', local_dir='/app'); \
    hf_hub_download(repo_id='chaehoyu/clock-vae-mono-100x-v1', filename='clock-vae-mono-100x-v1-1000epoch.pth', local_dir='/app'); \
    hf_hub_download(repo_id='matt3ounstable/dlib_predictor_recognition', filename='shape_predictor_68_face_landmarks.dat', local_dir='/app'); \
    hf_hub_download(repo_id='matt3ounstable/dlib_predictor_recognition', filename='dlib_face_recognition_resnet_model_v1.dat', local_dir='/app')"

# 애플리케이션 파일 복사
COPY main.py .
COPY clock_model_color.py .
COPY clock_model_mono.py .
COPY face_embedding.py .

# main.py 실행
CMD ["fastapi", "run", "main.py"]
