FROM python:3.11-slim

RUN apt update \
    && apt install -y --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/modele_CNN_commente.py ./modele_CNN_commente.py
COPY src/download.py ./download.py
COPY src/preprocess.py ./preprocess.py
COPY src/utils.py ./utils.py

CMD ["bash"]
