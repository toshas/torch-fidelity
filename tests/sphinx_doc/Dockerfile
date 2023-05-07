FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

RUN pip3 install --upgrade pip

RUN ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime
RUN groupadd --gid $GID $UNAME-group
RUN useradd --uid $UID --gid $GID -m -c $UNAME -s /bin/bash $UNAME
USER $UNAME
WORKDIR /work
