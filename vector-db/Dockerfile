FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libopenblas-dev
RUN mkdir -p /index && chmod 777 /index

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]