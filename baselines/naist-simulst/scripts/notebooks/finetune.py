import warnings
#warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKIN"] = "1"
from transformers import TrainingArguments, Trainer, BertForMaskedLM
import wandb

wandb.init(project="personal", entity="gojiteji")

import torch.nn as nn
from datasets import load_dataset
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    y = self.bert( input_ids=input_ids)
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

model = accmBERT().to('cuda')


train=load_dataset('ahclab/mBERT-acceptability-filtering-data', split='train',use_auth_token=True)
valid=load_dataset('ahclab/mBERT-acceptability-filtering-data', split='valid',use_auth_token=True)

training_args = TrainingArguments(
    output_dir='/ahc/work2/koki-tan/fine-tune3',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    label_names=['labels'],##Pytorchの型に基づく、データセットのカラム名ではない
    metric_for_best_model="accuracy",
    #auto_find_batch_size=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    report_to='wandb',
    learning_rate=5e-7,
    #resume_from_checkpoint="/ahc/work2/koki-tan/finetune2/checkpoint-12739"
)

def compute_metrics(pred):
  labels = pred.label_ids
  preds =list(map(lambda x: 1 if x[0] >0.5 else 0,pred.predictions))
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
      'f1': f1,
      'precision': precision,
      'recall': recall
  }
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    compute_metrics=compute_metrics
)
trainer.train()
#model.push_to_hub("ahclab/mBERT-acceptability-filtering")
