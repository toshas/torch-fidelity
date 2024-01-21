FROM nvcr.io/nvidia/pytorch:23.12-py3
# Supports torch>=1.11.0

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

COPY utils utils
RUN python3 utils/util_dump_dataset_as_images.py cifar10-train /tmp/cifar10-train-5000 -l 5000 && \
    python3 utils/util_dump_dataset_as_images.py cifar10-valid /tmp/cifar10-valid-5000 -l 5000

RUN pip3 install --upgrade pip
RUN pip3 uninstall -y torch torchvision torchtext torchdata torch-tensorrt pytorch-quantization
RUN pip3 freeze | grep torch
RUN pip3 install numpy pillow tqdm

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /work
