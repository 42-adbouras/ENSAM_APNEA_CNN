FROM python:3.11-slim

RUN apt update \
    && apt install -y --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/model.py ./model.py
COPY src/test.py ./test.py
COPY src/download.py ./download.py
COPY src/preprocess.py ./preprocess.py
COPY src/utils.py ./utils.py

CMD ["bash"]
