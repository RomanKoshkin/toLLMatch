## フィルター系コード
元論文
https://aclanthology.org/2020.acl-main.756/


想定されるデータセットは改行で分けられたテキストファイルです。
**拡張子は言語コード**である必要があります。
言語コードはsourceとtargetで重複してはいけません。

### simple_filering.py

 sourceとtargetの長さの比が1.3を超えるものをフィルタリングします。
 **このフィルタのみ、トークン化済みのデータを入力してください**

 ```bash
 python3 simple_filtering.py -s  train.en -t train.de -o ./
 ```
### LID_filtering.py

 言語コード(LID)を推定して、データと同じかどうかによりフィルタリングします。
 対応している言語は[fasttextの公式Doc](https://fasttext.cc/docs/en/language-identification.html)を参照してください。
```bash
python3 LID_filtering.py -s  train.en -t train.de -o ./ -path lid.bin
```

### domain_filtering.py

ドメインの違いを判定して除去するフィルタ。

対応言語は、targetが中国語、ドイツ語、日本語のみ。
    
Non domainの計算には、上記言語が比較的Noisyとなるmbartを使用（これがうまく動くかはわからない）。

大規模言語モデルを元にパープレキシティを出しているため、実行に時間がかかりま。(IWSLT2022/en-de、ahcgtb01で2時間ほど。)、GPUを並列で使用するため、GPUが使用可能なマシンでのみ実行可能。


ドメイン判定の上限閾値として、`t_cutoff`、下限閾値として`t_cutoff`が存在。閾値がどのくらいがいいかは分からないため、デフォルトでは閾値でクリーニングする前のテーブルをcsvで書き出す。

先に300個ぐらいのデータで走らせてみて、csvを見てみて閾値を決めるとよさそう。

実行してる感じ、GPUが多くても早いとは限らなさそう？

#### pandarallはGPUの利用に対応していないため、ライブラリに以下の変更を行う。
    
pandarallライブラリのインストール先である、`{インストールパス}/python3.9/site-packages/pandarallel/core.py`に
```Python3
import multiprocessing
```
    
を削除し、以下を追記

```Python3
import torch.multiprocessing as multiprocessing
if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)
```
    
pandaralleライブラリのインストール先である、`{インストールパス}/python3.9/site-packages/pandarallel/core.py`に
    
```Python3
import multiprocessing
```
    
を削除し、以下を追記し、
    
```Python3
import torch.multiprocessing as multiprocessing
if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)
```
`CONTEXT = multiprocessing.get_context("spawn" if ON_WINDOWS else "fork")`を、`CONTEXT = multiprocessing.get_context("spawn")`に変更

実行サンプル
```bash
python3 domain_filtering.py -s  train.en -t train.de -o ./ --save_rawdata True  --save_rawdata_path ./ --gpu 0 1 2
```
あんまりGPUの数で処理の負荷の違いが分からない気もする...

`CUDA error: out of memory`が出る場合は、--gpuで選択するGPUを減らしてみてください。

### acceptability_filtering.py

２言語間の質を多言語モデルの埋め込みを用いて測る。

**必ず`-s (--source)`が英語に、`-t (--target)がドイツ語 or 日本語 or 中国語になるようにしてください。**
    
閾値がどのくらいがいいかは分からないため、デフォルトでは閾値でクリーニングする前のテーブルをcsvで書き出す。
```bash
python3 acceptability_filtering.py -s  train.en -t train.de -o ./  -p {PATH to pytorch_model.bin} -b 16
```
    
コードとしては，0(not accept), 1(accept)で判定
    
   
モデルの入力は次のようにして加工
```Python
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
    src_tokens=src_tokens
    tgt_tokens=tgt_tokens
    ids=[tokenizer.cls_token_id] +src_tokens+[tokenizer.sep_token_id]+tgt_tokens
    ids_512=ids+[0]*(512-len(ids))
    return ids_512
```
### roberta_acceptability-enja.py

日本語と英語をkfttデータを用いてfine-tuneした．
flax実装のため，速いが，GPUはメモリが大きめなものが必要

実行コード
以下はバージョンに注意が必要なライブラリ
```
pip install --upgrade "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.6.0
pip install transformers[sentencepiece]
```

実行方法 ⚠GPUは一枚のみ対応
```Python
python3 roberta_acceptability-enja.py -s data.en -t data.ja --gpu 0
```

