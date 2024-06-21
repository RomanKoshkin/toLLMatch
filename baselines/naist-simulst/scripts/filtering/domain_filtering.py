import argparse
import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import warnings
from pandarallel import pandarallel

warnings.filterwarnings("ignore")
tqdm.pandas()

def lid_filter(args):
    src=args.source
    tgt=args.target
    out_dir=args.out_dir
    gpunumber=args.gpu
    if(args.save_rawdata):
        if args.save_rawdata_path==None:
            print("specify file to save raw table or use --save_rawdata False")
            exit()
        else:
            rawdata_dir=args.save_rawdata_path

    with open(src) as f:
        arr_src = [line.strip() for line in f]
    with open(tgt) as f:
        arr_tgt = [line.strip() for line in f]

    _,src_code= os.path.splitext(src)
    _,tgt_code= os.path.splitext(tgt)
    df_src=pd.DataFrame(arr_src,columns=[ src_code ])
    df_tgt=pd.DataFrame(arr_tgt,columns=[ tgt_code ])
    df=df_src.join(df_tgt)
    df['in_ppl_src'] = None
    df['out_ppl_src'] = None
    df['in_ppl_tgt'] = None
    df['out_ppl_tgt'] = None
    df['f_src_raw'] = None
    df['f_tgt_raw'] = None
    df['New_ID'] = range(len(df))
    
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if(device=="cpu"):
        print("ONLY GPU IS SUPPORTED")
        exit()
    print("using",device)


    print("loading model...")
    lang_code=tgt_code[1:]
    if lang_code=="de":
        from transformers import AutoTokenizer, AutoModelWithLMHead
        in_tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
        in_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2") 

    elif lang_code=="zh":
        from transformers import BertTokenizer, TFGPT2LMHeadModel
        from transformers import TextGenerationPipeline
        in_tokenizer = BertTokenizer.from_pretrained("mymusise/EasternFantasyNoval")
        in_model = TFGPT2LMHeadModel.from_pretrained("mymusise/EasternFantasyNoval")

    elif lang_code=="ja":
        from transformers import T5Tokenizer, AutoModelForCausalLM
        in_tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        in_tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        in_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    else:
        print("Supported language is only de, zh, and ja")
        exit()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    out_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    out_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")



    def calcurate_ppl(row,code,model,tokenizer):
        cnt=row["New_ID"]
        n=gpunumber[cnt%len(gpunumber)]
        cnt+=1
        model=model.cuda(n)
        txt=row[code]
        max_length = 1024
        stride = 1024
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encodings = tokenizer(txt,padding=True, return_tensors="pt")
        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda(n)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids.cuda(n), labels=target_ids.cuda(n))
            neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood.item())
        ppl = 2.71828182846**(sum(nlls) / end_loc)
        return ppl

    print("calcurating...")
    pandarallel.initialize(progress_bar=True,nb_workers=len(gpunumber),use_memory_fs=False)
    print("step 1/4")
    df['in_ppl_src']= None
    df['in_ppl_src'] = df.parallel_apply(calcurate_ppl,args=(src_code,in_model,in_tokenizer),axis=1)
    print("step 2/4")
    df['out_ppl_src']= None
    df['out_ppl_src'] = df.parallel_apply(calcurate_ppl,args=(src_code,out_model,out_tokenizer),axis=1)
    print("step 3/4")
    df['in_ppl_tgt']= None
    df['in_ppl_tgt'] = df.parallel_apply(calcurate_ppl,args=(tgt_code,in_model,in_tokenizer),axis=1)
    print("step 4/4")
    df['out_ppl_tgt']= None
    df['out_ppl_tgt'] = df.parallel_apply(calcurate_ppl,args=(tgt_code,out_model,out_tokenizer),axis=1)

    df['f_src_raw']=df.progress_apply(lambda x:x['in_ppl_src']/x['out_ppl_src'],axis=1)
    df['f_tgt_raw']=df.progress_apply(lambda x:x['in_ppl_tgt']/x['out_ppl_tgt'],axis=1)

    clip_cut=lambda x,col: 1 if args.t_clip  < x[col] else False if args.t_cutoff > x[col] else True
    df['pplsrc']=df.progress_apply(clip_cut,col='f_src_raw',axis=1)
    df['ppltgt']=df.progress_apply(clip_cut,col='f_tgt_raw',axis=1)
    if args.save_rawdata:
        print("saving raw data...")
        df.to_csv(rawdata_dir+"rawtable.csv")
        print("saved!")
    new_df=df[df['pplsrc'] & df['ppltgt']]
    np.savetxt(out_dir+'/cleaned'+src_code, new_df[src_code], fmt='%s')
    np.savetxt(out_dir+'/cleaned'+tgt_code, new_df[tgt_code], fmt='%s')

if __name__ == "__main__":
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
        "--out_dir",
        "-o",
        type=str,
        required=True,
        help="path to the save cleaned txt",
    )

    parser.add_argument(
        "--gpu",
        "-g",
        nargs='+',
        type=int,
        required=True,
        help="gpu number to use",
    )


    parser.add_argument(
        "--save_rawdata",
        "-sr",
        type=bool,
        help="save row (no cut and clip) table",
        default=True
    )

    parser.add_argument(
        "--save_rawdata_path",
        "-sp",
        type=str,
        help="path to the save raw (no cut and clip) table",
        default=None
    )
    parser.add_argument(
        "--t_clip",
        "-clip",
        type=float,
        help="clip threadshold",
        default=1.0
    )
    parser.add_argument(
        "--t_cutoff",
        "-cut",
        type=float,
        help="cutoff threadshold",
        default=0.2
    )

    args = parser.parse_args()

    lid_filter(args)