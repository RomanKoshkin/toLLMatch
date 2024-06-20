#!/bin/bash

pip install --upgrade pip
pip install -U git+https://github.com/huggingface/transformers.git@e03a9cc
pip install python-dotenv
pip install numpy==1.23.5
pip install nltk==3.8.1
pip install termcolor==2.4.0
pip install flash-attn==2.5.7
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.8.3
cd ../whisper-jax && pip install -e . && cd ../scripts
cd ../SimulEval && pip install -e . && cd ../scripts
pip install vllm==0.4.1
pip install fastapi==0.111.0
pip install fastapi-cli==0.0.2
pip install editdistance==0.8.1
pip install chardet==5.2.0
