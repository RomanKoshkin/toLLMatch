# build frontend with node
FROM node:20-alpine AS frontend
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY streaming-react-app .
RUN \
    if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then yarn global add pnpm && pnpm i --frozen-lockfile; \
    else echo "Lockfile not found." && exit 1; \
    fi

RUN npm run build

# build backend on CUDA 
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS backend
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_MAJOR=20

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    sox libsox-fmt-all \
    # gradio dependencies \
    ffmpeg \
    # fairseq2 dependencies \
    libjpeg8-dev \
    libpng-dev \
    libsndfile-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER root
RUN ln -s /usr/lib/x86_64-linux-gnu/libsox.so.3 /usr/lib/x86_64-linux-gnu/libsox.so
# install older versions libjpeg62-turbo and libpng15
RUN wget http://ftp.us.debian.org/debian/pool/main/libj/libjpeg-turbo/libjpeg62-turbo_2.1.5-2_amd64.deb && \
    dpkg -i libjpeg62-turbo_2.1.5-2_amd64.deb && \
    rm libjpeg62-turbo_2.1.5-2_amd64.deb
RUN wget https://master.dl.sourceforge.net/project/libpng/libpng15/1.5.30/libpng-1.5.30.tar.gz && \
    tar -xvf libpng-1.5.30.tar.gz && cd libpng-1.5.30 && ./configure && make && make install && cd .. && rm -rf libpng-1.5.30.tar.gz libpng-1.5.30
    
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ARG PYTHON_VERSION=3.10.12
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

COPY --chown=user:user ./seamless_server ./seamless_server
# change dir since pip needs to seed whl folder
RUN cd seamless_server && \
    pip install fairseq2 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.1.1/cu118 && \
    pip install --no-cache-dir --upgrade -r requirements.txt
COPY --from=frontend /app/dist ./streaming-react-app/dist

WORKDIR $HOME/app/seamless_server
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \ 
    huggingface-cli login --token $(cat /run/secrets/HF_TOKEN) || echo "HF_TOKEN error" && \
    huggingface-cli download meta-private/SeamlessExpressive pretssel_melhifigan_wm-final.pt  --local-dir ./models/Seamless/ || echo "HF_TOKEN error" && \
    ln -s $(readlink -f models/Seamless/pretssel_melhifigan_wm-final.pt) models/Seamless/pretssel_melhifigan_wm.pt || true;

USER user
RUN ["chmod", "+x", "./run_docker.sh"]
CMD ./run_docker.sh

