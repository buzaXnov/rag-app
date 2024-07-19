FROM python:3.9-slim

WORKDIR /app

COPY local_data/ /app/local_data
COPY faiss_index /app/faiss_index
COPY services/ /app/services
COPY config.py /app/config.py
COPY utils.py /app/utils.py
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update -y && apt-get install -y curl

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
