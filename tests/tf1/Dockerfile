FROM nvcr.io/nvidia/pytorch:19.02-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install \
        torch==1.10.1 \
        torchvision==0.11.2

RUN pip install --ignore-installed wrapt && \
    pip install \
        scipy \
        imageio \
        scikit-learn \
        tensorflow-gpu==1.14 \
        tensorflow-determinism==0.3.0

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /work
