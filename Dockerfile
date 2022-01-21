FROM pytorch/pytorch:1.7-cuda10.1-cudnn7-runtime
RUN pip install torchvision=0.5.0 scikit-image=0.18.2 pandas=1.3.1 czifile=2019.7.2 nd2reader=3.3.0
RUN apt-get update && apt-get install curl -y

COPY . /app
RUN mkdir -p /app/models/zeisscyto
RUN curl -f -o /app/models/zeisscyto/zeisscyto.pt https://mcmicro.s3.amazonaws.com/models/WGA-maskRCNN/zeisscyto.pt
RUN curl -f -o /app/models/zeisscyto/cocomodel.pt https://mcmicro.s3.amazonaws.com/models/WGA-maskRCNN/cocomodel.pt
