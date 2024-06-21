import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
tqdm.pandas()
import ast
import warnings
from transformers import Trainer,TrainingArguments
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKIN"] = "1"
from transformers import BertForMaskedLM,BertTokenizer

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class accmBERT(nn.Module):
  def __init__(self):
    super(accmBERT,self).__init__()
    self.dim=768*512
    self.channels=64
    self.kernel_size=64
    self.lout=int(((self.dim-(self.kernel_size-1)-1)/1)+1)

    self.bert=BertForMaskedLM.from_pretrained("bert-base-multilingual-uncased",output_hidden_states=True, output_attentions=False)
    self.conv=nn.Conv1d(1,self.channels,self.kernel_size)
    self.dropout = nn.Dropout(0.2) 
    self.linear = nn.Linear(self.lout*self.channels,1) # load and initialize weights
    self.out = nn.Sigmoid()

  def forward(self, input_ids, attention_mask=None,labels=None):
    y = self.bert(input_ids=input_ids)
    y = self.conv(y["hidden_states"][-1].view(-1,1,self.dim))
    y = self.dropout(y.view(-1,self.lout*self.channels))
    y = self.linear(y)
    y = self.out(y)
    
    loss=None
    if labels is not None:
      loss_fct = nn.BCELoss()
      loss = loss_fct(y, labels.view(-1,1).float())


    return SequenceClassifierOutput(
            logits=y,
            loss=loss,
        )



def acc_filter(args):
    src=args.source
    tgt=args.target
    out_dir=args.out_dir
    params_dir=args.params
    threshold=args.threshold
    batch_size=args.batchsize

    with open(src) as f:
        arr_src = [line.strip() for line in f]
    with open(tgt) as f:
        arr_tgt = [line.strip() for line in f]

    _,src_code= os.path.splitext(src)
    _,tgt_code= os.path.splitext(tgt)
    if src_code is tgt_code:
        print('Don\'t use same language code.')
        exit()
    if src_code !=  ".en":
        print("src must be english")
        exit()
    df_src=pd.DataFrame(arr_src,columns=[ src_code ])
    df_tgt=pd.DataFrame(arr_tgt,columns=[ tgt_code ])
    df=df_src.join(df_tgt)
    df['ids'] = None
    df['acceptable_raw'] = None


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    # tokenizer = AutoTokenizer.from_pretrained(params_dir,local_files_only=True)
    #全データtokenizeする
    
    
    def preprocess(x):
        src_tokens=tokenizer(x[src_code], add_special_tokens=False, padding="max_length", max_length=256,truncation=True)['input_ids']
        tgt_tokens=tokenizer(x[tgt_code], add_special_tokens=False, padding="max_length", max_length=256,truncation=True)['input_ids']
        if 0 in  src_tokens:
            src_tokens=src_tokens[:src_tokens.index(0)]
        else:
            src_tokens=src_tokens[:-1]
        if 0 in  tgt_tokens:
            tgt_tokens=tgt_tokens[:tgt_tokens.index(0)]
        else:
             tgt_tokens=tgt_tokens[:-1]
        src_tokens=src_tokens#なぜか最後にsepがつくけど、そう学習してしまったのでそのままで
        tgt_tokens=tgt_tokens
        ids=[tokenizer.cls_token_id] +src_tokens+[tokenizer.sep_token_id]+tgt_tokens
        ids_512=ids+[0]*(512-len(ids))
        return ids_512
    df['ids']=df.parallel_apply(preprocess,axis=1)

    # df.to_csv("/ahc/work2/ryuta-is/reexpriment/finetune-mbart/acc_filtered/tmp_train.csv")
    
    # exit()
    # df.pd.read_csv()でidsがstr型になってしまい、それをast.literal_eval(x)でlistに直してるから、def preprocessして推論まで実行するときはcsvに書き出す必要あり。
    # df = pd.read_csv("/ahc/work2/ryuta-is/reexpriment/finetune-mbart/acc_filtered/tmp.csv")

    #バッチ化して実行
    print('working') 
    ids= df['ids'].to_numpy()
    print('working2')
    model = accmBERT().to('cuda')
    # model=AutoModelForSeq2SeqLM.from_pretrained(params_dir,local_files_only=True)
    print('working3')
    model.load_state_dict(torch.load(params_dir))
    print('working4')
    model.eval()
    rates=[]

    
    # for b in tqdm(range(0,len(df['ids']),batch_size)):
    #     batch=None
    #     try:
    #         batch=np.array([ ast.literal_eval(txt) for txt in ids[b:b+batch_size]])
    #         batch=torch.Tensor(batch).long().to("cuda")
    #     except Exception as e:
    #         print("------------------------------------------------------------")
    #         print(e)
    #         print(b+batch_size)
    #         print(ids[b:b+batch_size])
    #     tmp=model(input_ids=batch.reshape(-1,512)).logits.squeeze(1).cpu().detach().numpy()
    #     for t in tmp:
    #         rates.append(t)
    
    # import pdb; pdb.set_trace()
    
    # ids = list(map(lambda x:ast.literal_eval(x), ids))
    # ids = [id for id in ids if len(id) == 512]
    data= {"input_ids" : ids} 
    # df.read_csv()でデータを読み込むときはこっち↓を使用
    # data= {"input_ids" : list(map(lambda x:ast.literal_eval(x) , tqdm(ids)))}
    # import pdb; pdb.set_trace()
    # del ids
    print("working5")
    from datasets import Dataset
    data=Dataset.from_dict(data)
    # longest_input_ids = max(map(len,tqdm(data['input_ids'])))
    
    """
    data={'input_ids': [101, 10346,.......,0],[],[]}
    """
    print("working6")
    #from torch.utils.data import  DataLoader 
    #data=DataLoader(data,batch_size=batch_size)
    print("working7")
    training_args = TrainingArguments(
        auto_find_batch_size=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        output_dir=out_dir,
        resume_from_checkpoint=True,
        per_device_eval_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size
    )

    # tmp=model(input_ids=data)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
    )
  
    # import pdb; pdb.set_trace()
    result=trainer.predict(data)#,description="prediction")
    print(f'result:{result}')
    # print(f'result shape{result.shape}')
    rates_torch=result.predictions

    for t in tqdm(rates_torch):
        rates.append(t)


    df['acceptable_raw']=rates
    new_df=df[df['acceptable_raw'] >= threshold]

    print("saving cleaned"+src_code)
    np.savetxt(out_dir+'/cleaned'+src_code, new_df[src_code].to_numpy(), fmt='%s')
    print("saving cleaned"+tgt_code)
    np.savetxt(out_dir+'/cleaned'+tgt_code, new_df[tgt_code].to_numpy(), fmt='%s')
    print("saving raw table "+tgt_code)
    np.savetxt(out_dir+'/raw_data.txt', df["acceptable_raw"].to_numpy(), fmt='%s')


if __name__ == "__main__":
    if(torch.cuda.is_available()==False):
        print("GPU not detected. Please use GPU server.")
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="path to the source txt",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="path to the target txt",
    )
    
    parser.add_argument(
        "--params",
        "-p",
        type=str,
        required=True,
        help="path to the params",
    )

    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        required=True,
        help="path to the save cleaned txt",
    )

    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        required=True,
        help="inference batch size",
    )

    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        help="acceptability rate threadshold",
        default=0.50
    )

    args = parser.parse_args()

    acc_filter(args)
