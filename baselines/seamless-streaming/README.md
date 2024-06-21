---
title: Seamless Streaming
emoji: 📞
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
suggested_hardware: t4-small
models:
 - facebook/seamless-streaming
---

# Seamless Streaming demo
## Running on HF spaces
You can simply duplicate the space to run it.

## Running locally
### Install backend seamless_server dependencies

> [!NOTE]
> Please note: we *do not* recommend running the model on CPU. CPU inference will be slow and introduce noticable delays in the simultaneous translation.

> [!NOTE]
> The example below is for PyTorch stable (2.1.1) and variant cu118. 
> Check [here](https://pytorch.org/get-started/locally/) to find the torch/torchaudio command for your variant. 
> Check [here](https://github.com/facebookresearch/fairseq2#variants) to find the fairseq2 command for your variant.

If running for the first time, create conda environment and install the desired torch version. Then install the rest of the requirements:
```
cd seamless_server
conda create --yes --name smlss_server python=3.8 libsndfile==1.0.31
conda activate smlss_server
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fairseq2 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.1.1/cu118
pip install -r requirements.txt
```

### Install frontend streaming-react-app dependencies
```
conda install -c conda-forge nodejs
cd streaming-react-app
npm install --global yarn
yarn
yarn build  # this will create the dist/ folder
```


### Running the server

The server can be run locally with uvicorn below.
Run the server in dev mode:

```
cd seamless_server
uvicorn app_pubsub:app --reload --host localhost
```

Run the server in prod mode:

```
cd seamless_server
uvicorn app_pubsub:app --host 0.0.0.0
```

To enable additional logging from uvicorn pass `--log-level debug` or `--log-level trace`.


### Debuging

If you enable "Server Debug Flag" when starting streaming from the client, this enables extensive debug logging and it saves audio files in /debug folder. 