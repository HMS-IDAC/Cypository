FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN pip install torchvision scikit-image pandas czifile nd2reader
RUN apt-get update && apt-get install curl -y

COPY . /app
RUN mkdir -p /app/models/zeisscyto
RUN curl -f -o /app/models/zeisscyto/zeisscyto.pt https://mcmicro.s3.amazonaws.com/models/WGA-maskRCNN/zeisscyto.pt
RUN curl -f -o /app/models/zeisscyto/cocomodel.pt https://mcmicro.s3.amazonaws.com/models/WGA-maskRCNN/cocomodel.pt