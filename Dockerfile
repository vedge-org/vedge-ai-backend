# 베이스 이미지
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# Hugging Face Hub에서 .pth 파일 다운로드
RUN python -c "\
    from huggingface_hub import hf_hub_download;\
    hf_hub_download(repo_id='chaehoyu/clock-vae-color-140x-v1', filename='clock-vae-color-140x-v1-500epoch.pth', local_dir='/app');\
    hf_hub_download(repo_id='chaehoyu/clock-vae-mono-100x-v1', filename='clock-vae-mono-100x-v1-1000epoch.pth', local_dir='/app')"

# 애플리케이션 파일 복사
COPY main.py .

# main.py 실행
CMD ["python", "main.py"]
