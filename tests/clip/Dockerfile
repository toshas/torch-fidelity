FROM nvcr.io/nvidia/pytorch:20.11-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install -U torch

RUN pip3 install \
    ftfy \
    regex \
    tqdm \
    git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 \
    clean-fid==0.1.35

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /work
