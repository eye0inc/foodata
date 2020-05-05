FROM "tensorflow/tensorflow:latest-gpu-py3"

COPY . /datahub

WORKDIR /datahub

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

