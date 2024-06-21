import jax
import jax.numpy  as np
from jax.lib import xla_bridge
import flax
from functools import partial
import optax
from tqdm import tqdm
from transformers import FlaxXLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from flax.training.train_state import TrainState
import os
import warnings
import argparse
import pandas as pd

warnings.filterwarnings("ignore")
num_devices: int=1
os.environ["TZ"]="Asia/Tokyo"
os.environ["WANDB_DISABLED"]="true"
os.environ["DATASETS_VERBOSITY"]="error"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'



batch_size :int = 1
SEED :int= 0
basesize:int=130
learning_rate :float = 1e-5
epochs:int=64




tokenizer=XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large",output_hidden_states=False, output_attentions=False,cache_dir="/ahc/work2/koki-tan/.cache/huggingface")




class DataCollatorWithPadding:
    def __init__(self,tokenizer,return_tensors="np",basesize=128):
        self.tokenizer=tokenizer
        self.return_tensors=return_tensors
        self.basesize=basesize
    def __call__(self,batch):
        maxlen = max([len(x["input_ids"]) for x in batch])
        pad_to=self.basesize
        while(pad_to<maxlen):
            pad_to=2*pad_to
        padded_data=tokenizer.pad(batch,pad_to_multiple_of=pad_to,return_tensors=self.return_tensors)
        batch = {k: np.array(v) for k, v in padded_data.items()}
        return batch


#for pmap
@partial(jax.pmap, axis_name="num_devices")
def eval_step(state, input_ids,attention_mask):
    logits = state.apply_fn(params=state.params,input_ids=input_ids,attention_mask=attention_mask).logits.squeeze(-1)
    preds = logits.flatten()
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="engltish txt",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="target txt",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        nargs='+',
        type=int,
        required=True,
        help="gpu number to use",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)[1:-1]
    print("using gpus:",os.environ["CUDA_VISIBLE_DEVICES"])


    file1 = open(args.source, 'r')
    Lines1 = file1.readlines()
    Lines1 = [line.replace('\n', '') for line in Lines1]
    file2 = open(args.target, 'r')
    Lines2 = file2.readlines()
    Lines2 = [line.replace('\n', '') for line in Lines2]

    data = []
    for i in range(len(Lines1)):
        en=tokenizer(Lines1[i],add_special_tokens=False)['input_ids']
        ja=tokenizer(Lines2[i],add_special_tokens=False)['input_ids']
        tokens=[tokenizer.cls_token_id] +en+[tokenizer.sep_token_id]+ja+[tokenizer.eos_token_id]
        data.append(tokens)
    df = pd.DataFrame({"input_ids":data})
    evaldata = Dataset.from_pandas(df)
        
    collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors="np",basesize=basesize)
    debug_dataloader = DataLoader(evaldata, batch_size=batch_size*num_devices,collate_fn=collator, shuffle=False)
    scheduer = optax.linear_schedule(learning_rate, 0.0, 100)
    tx = optax.adamw(learning_rate=scheduer)

    model = FlaxXLMRobertaForSequenceClassification.from_pretrained("/ahc/work2/koki-tan/flaxreward-enja/epoch-2/")
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx)
    #for pmap
    state = flax.jax_utils.replicate(state)


    y_pred=np.array([])
    for batch in tqdm(debug_dataloader):
        input_ids = batch["input_ids"].reshape(num_devices,batch_size,-1)
        attention_mask = batch["attention_mask"].reshape(num_devices,batch_size,-1)
        preds = eval_step(state,input_ids,attention_mask)
        y_pred=np.append(y_pred,preds)
    
    cleaned_data_en = []
    cleaned_data_ja = []
    for i in range(len(Lines1)):
        if y_pred[i]>0:
            cleaned_data_en.append(Lines1[i])
            cleaned_data_ja.append(Lines2[i])
    with open('out.en', 'w') as fp:
        for item in cleaned_data_en:
            fp.write("%s\n" % item)
    with open('out.ja', 'w') as fp:
        for item in cleaned_data_ja:
            fp.write("%s\n" % item)
