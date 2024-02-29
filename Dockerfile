FROM nvcr.io/nvidia/pytorch:22.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y libpgm-dev net-tools

COPY zeromq-4.3.4.tar.gz .
RUN tar zxvf zeromq-4.3.4.tar.gz
WORKDIR zeromq-4.3.4
RUN ./configure --with-pgm && make && make install

RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install -I --no-binary=:all: pyzmq
RUN pip3 install -U gym[atari] gym[accept-rom-license] prometheus_client wandb redis pymongo \
    Deprecated einops kornia pysc2
RUN pip3 install opencv-python==4.5.4.60
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /distributed_marl/

WORKDIR /distributed_marl
RUN bash scripts/build_muzero_mcts