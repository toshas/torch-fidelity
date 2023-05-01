FROM nvcr.io/nvidia/pytorch:20.11-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install \
    scipy \
    imageio \
    scikit-learn \
    tensorflow-gpu==1.15.2 \
    framework-reproducibility

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /work
