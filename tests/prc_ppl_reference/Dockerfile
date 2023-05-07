FROM nvcr.io/nvidia/pytorch:20.11-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install -U torch
RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME

RUN git clone https://github.com/toshas/torch-fidelity-stylegan2-ada.git
RUN git -C torch-fidelity-stylegan2-ada checkout a3a137d

WORKDIR /work
