FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py"]