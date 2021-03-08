FROM nvidia/cuda:9.2-base-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
     build-essential \
     cmake \
     python3-dev \
     python3-pip \
     python3-setuptools \
     curl \
     ca-certificates \
     python-qt4 \
     libjpeg-dev \
     zip \
     unzip

COPY recsys /home/user/recsys
COPY setup.py /home/user/

RUN pip3 install wheel
RUN pip3 install -e /home/user/

CMD [ "python3", "-m", "recsys.submission"]