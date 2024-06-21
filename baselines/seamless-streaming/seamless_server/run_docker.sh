# !/bin/bash
if [ -f models/Seamless/pretssel_melhifigan_wm.pt ] ; then
    export USE_EXPRESSIVE_MODEL=1;
fi
uvicorn app_pubsub:app --host 0.0.0.0 --port 7860