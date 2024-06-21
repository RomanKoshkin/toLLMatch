#!/bin/bash

pip install --upgrade pip
pip install numpy==1.23.5
pip install python-dotenv==1.0.1
pip install nltk==3.8.1
pip install termcolor==2.4.0
cd ../SimulEval && pip install -e . && cd ../scripts
pip install fastapi-cli==0.0.2
pip install editdistance==0.8.1
pip install chardet==5.2.0
pip install openai==1.23.1
pip install -U git+https://github.com/huggingface/transformers.git@e03a9cc
pip install msgpack==1.0.8
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
cd ../whisper-jax && pip install -e . && cd ../scripts
pip install numpy==1.23.5
pip install vllm==0.4.1
pip install numpy==1.23.5
pip install accelerate==0.31.0
pip intall flax==0.8.4
pip install numpy==1.23.5
pip install -U jax[cuda12]==0.4.28  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U jaxlib==0.4.28+cuda12.cudnn89  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flash-attn==2.5.8