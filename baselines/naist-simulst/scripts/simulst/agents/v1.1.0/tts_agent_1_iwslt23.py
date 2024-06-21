import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))
from sub2yomi import *
from base_pron import *
from pwg import *
import torch
from scipy.io import wavfile
from argparse import ArgumentParser

try:
    from simuleval.utils import entrypoint
    from simuleval.agents import TextToSpeechAgent, AgentStates
    from simuleval.agents.actions import Action, ReadAction, WriteAction
    from simuleval.data.segments import Segment, TextSegment, EmptySegment, SpeechSegment
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


def make_test_input(hp_sub2yomi,vocabs):
  with open("./test.jsut.bpe.txt","r") as f:
    text=f.read().strip().split("\n")
  text=[t.split(" ") for t in text]
  tmp=[]
  txts=[]
  for txt in text:
    """
    if txt[0]=="\u2581":
      txt[0]="<s>"
    else:
      txt=["<s>"]+txt
    """
    if txt[0]=="\u2581":
      txt=txt[1:]
    txt=txt+["</s>"]
    tmp.append(np.asarray([vocabs[0][char] if char in vocabs[0] else vocabs[0]["<unk>"] for char in txt],dtype=np.int32))
    txts.append(txt)
  return tmp,txts


def detect_pos(yomi_queue,pos_queue):
  if all([p_q=="<ww>" for p_q in pos_queue]):
    return None,yomi_queue,pos_queue #input to tts,queue for yomi, queue for pos
  elif "</s>" in yomi_queue:
    return yomi_queue,[],[]
  tmp_yomi_out=[]
  yomi_out=[]
  idx=0 #for pop to queue
  for y_q,p_q in zip(yomi_queue,pos_queue):
    tmp_yomi_out+=y_q
    if p_q!="<ww>":
      yomi_out+=tmp_yomi_out
      tmp_yomi_out=[]
      idx=len(yomi_out)
  yomi_q_out=yomi_queue[idx:]
  pos_q_out=pos_queue[idx:]
  return yomi_out,yomi_q_out,pos_q_out


def remove_strings(text, exclude_list):
    for exclude_str in exclude_list:
        text = text.replace(exclude_str, "")
    return text


class TTSAgentStates(AgentStates):

    def __init__(self,sub2yomi_model_path:str,yomi2tts_model_path:str,tts2wav_model_path:str,sub2yomi_dict_path:str,yomi2tts_dict_path:str) -> None:
        self.devices="cuda" if torch.cuda.is_available() else "cpu" #cpu or gpu
        self.vocabs=sub2yomi.load_vocab(sub2yomi_dict_path) #vocabs for subword2yomi
        self.inv_vocabs=[{i:e for i,e in enumerate(self.vocabs[idx])} for idx in range(len(self.vocabs))] #inversed vocabs
        self.yomi_model=sub2yomi.model_load(sub2yomi.hp_sub2yomi,sub2yomi_model_path,self.vocabs) #suword2yomi model
        self.model1=sub2yomi.sub2yomi(self.yomi_model,self.devices,self.vocabs) #class of subword2yomi model
        self.model2=tts.tts(self.devices,tts.taco2load(yomi2tts_model_path),wait=3,dict_path=yomi2tts_dict_path,eos_seq=None) #load trained tacotron2 and class of yomi2feats model
        self.vocoder=pwg.builde_model() #model of feats2speech(vocoder)
        self.vocoder=pwg.load_model(self.vocoder,tts2wav_model_path) #load vocoder
        self.model3=pwg.PWG(self.devices,self.vocoder) #model of vocoder

    def reset(self) -> None:
        self.source = None
        self.source_ids = None
        self.source_already_finished = False

        # model1
        self.yomi_out = None  # output of model1
        self.model1.initialize()
        self.memory = None  # encoder's memory for decoder of model1
        self.decoder_in = [
            torch.from_numpy(np.array(self.vocabs[idx]["<s>"]))
            .type(torch.cuda.LongTensor).cuda().unsqueeze(0)
            for idx in range(1, len(self.vocabs))
        ]  # initial decoder input of model1
        self.yomi_queue = []
        self.pos_queue = []
        self.wait1_idx = 0  # wait count for model1

        self.model2.initialize()
        self.is_first = True  # flag for first input
        self.is_decode_first = True  # flag for first decode
        self.prev_memory2 = None
        # initial mel_inputs for model2
        self.mel_inputs = \
            torch.zeros((1,hp_taco2.num_mels,1),dtype=torch.float).to(self.devices)
        self.out_pos = []
        self.cat_inputs = None  # encoder input of model2
        
        return super().reset()

    def update_source(self, segment: Segment):
        """
        Update states from input segment
        Additionally update incremental states
        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished

        # filter out special symbols
        if not segment.is_empty:
            exclude_list = ["( 拍 手 )", "( 笑 )", "\u2581"]
            segment.content = remove_strings(segment.content, exclude_list)

        if self.source_finished:
            if self.source_already_finished:
                self.source = []
            else:
                if segment.is_empty:
                    self.source = ["</s>"]
                else:
                    self.source = segment.content.split() + ["</s>"]
                    if self.source[0] == "\u2581":
                        self.source = self.source[1:]
                self.source_already_finished = True
        elif not self.source_finished and not segment.is_empty:
            self.source = segment.content.split()
        else:
            self.source = []

        # convert punctuation marks to space
        self.source = [word if word not in ["。", "、"] else " " for word in self.source]

        self.source_ids = (
            np.asarray([self.vocabs[0][word] if word in self.vocabs[0] else self.vocabs[0]["<unk>"]
                for word in self.source], dtype=np.int32,
        ))


def reduce_commas(arr):
    new_arr = []
    for i, item in enumerate(arr):
        if item == "、" and i > 0 and arr[i-1] == "、":
            continue
        new_arr.append(item)
    while new_arr and new_arr[-1] == "、":
        new_arr.pop()
    return new_arr


class Subword2Yomi2SpeechAgent(TextToSpeechAgent):

    def __init__(self, args):
        super().__init__(args)

        self.args = args

    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--tts-lookahead",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--print_tts_input",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--sub2yomi_model_path",
            type=str,
            default=None
        )
        parser.add_argument(
            "--yomi2tts_model_path",
            type=str,
            default=None
        )
        parser.add_argument(
            "--tts2wav_model_path",
            type=str,
            default=None
        )
        parser.add_argument(
            "--sub2yomi_dict_path",
            type=str,
            default=None
        )
        parser.add_argument(
            "--yomi2tts_dict_path",
            type=str,
            default=None
        )

    def build_states(self):
        return TTSAgentStates(\
                self.args.sub2yomi_model_path,\
                self.args.yomi2tts_model_path,\
                self.args.tts2wav_model_path,\
                self.args.sub2yomi_dict_path,\
                self.args.yomi2tts_dict_path\
                )

    def policy(self) -> Action:
        if len(self.states.source) == 0:
            if self.states.source_finished:
                return WriteAction(
                    SpeechSegment(
                        content=[0.0] * 1000,
                        sample_rate=16_000,
                        finished=True,
                    ),
                    finished=True,
                )
            return ReadAction()
        else:
            if self.args.print_tts_input:
                print(
                    f"translation:\
                    {''.join(self.states.source).replace('</s>', '<eos>')}"
                )

            inputs = torch.from_numpy(self.states.source_ids).type(torch.cuda.LongTensor).cuda()
            yomi_out = []
            for input_ in inputs:
                if hp_sub2yomi.wait > self.states.wait1_idx:
                    enc, h, c = self.states.model1.encode(input_.unsqueeze(0))
                    self.states.memory = enc.unsqueeze(1) if self.states.memory is None \
                        else torch.cat((self.states.memory, enc.unsqueeze(1)), dim=1)
                    self.states.wait1_idx += 1
                    continue
                else:
                    if input_.data.item() == self.states.model1.yomi_eos_id:
                        self.states.model1.yomi_model.decoder.is_input_eos = True
                    enc, h, c = self.states.model1.encode(input_.unsqueeze(0))
                    self.states.memory = enc.unsqueeze(1) if self.states.memory is None \
                        else torch.cat((self.states.memory, enc.unsqueeze(1)), dim=1)
                    outs, _ = self.states.model1.decode(self.states.memory, h, c, self.states.decoder_in)
                    self.states.decoder_in = [outs[out_id][-1] for out_id in range(len(outs))]
                    self.states.yomi_queue += [
                        self.states.inv_vocabs[1][out[0].data.item()] \
                        if out.data.item() != self.states.model1.yomi_unk_id \
                        else "" for out in outs[0]
                    ]
                    self.states.pos_queue += [
                        self.states.inv_vocabs[2][out[0].data.item()] \
                        if out.data.item() != self.states.vocabs[2]["<unk>"] \
                        else "" for out in outs[1]
                    ]
            yomi_out, self.states.yomi_queue, self.states.pos_queue = \
                detect_pos(self.states.yomi_queue, self.states.pos_queue)

            # TTS
            if not yomi_out:
                return ReadAction()
            if len(yomi_out) > 0:
                # reduce commas
                yomi_out = reduce_commas(yomi_out)
                if self.states.is_first:
                    yomi_out = ["<s>"] + yomi_out
                    self.states.is_fist = False

            inputs = torch.from_numpy(self.states.model2.text2seq(yomi_out))\
                .unsqueeze(0).long().to(self.states.devices)
            if self.states.is_decode_first and not self.states.source_finished:

                n_lookahead = self.args.tts_lookahead
                if inputs.shape[-1] - n_lookahead > 0:
                    self.states.out_pos.append(inputs.shape[-1] - n_lookahead)
                else:
                    self.states.out_pos.append(1)  # <s>
                    #self.states.out_pos.append(inputs.shape[-1])
                memory2 = self.states.model2.encode(inputs)
                self.states.is_decode_first = False

            else:

                self.states.out_pos.append(
                    inputs.shape[-1] if len(self.states.out_pos) == 0
                    else self.states.out_pos[-1] + inputs.shape[-1]
                )
                memory2 = self.states.model2.encode(inputs)

            if self.states.model2.eos_seq in inputs:
                mel_out, post_mel_out, acct, stop = self.states.model2.decode(
                    memory2, self.states.mel_inputs, -1
                )
                wav = self.states.model3.synthesis(post_mel_out)
            else:
                mel_out, post_mel_out, acct, stop = self.states.model2.decode(
                    memory2, self.states.mel_inputs, self.states.out_pos[-1]
                )
                wav = self.states.model3.synthesis(post_mel_out)
                self.states.mel_inputs = mel_out[:,:,-1].unsqueeze(2) #next tacotron2's inputs

            wav = wav.astype(np.float32) / 32768  # ad-hoc
            max_length = 30  # in seconds
            wav = wav[:max_length*16_000]
            return WriteAction(
                SpeechSegment(
                    content=wav.tolist(),
                    sample_rate=16_000,
                    finished=self.states.source_finished,
                ),
                finished=self.states.source_finished,
            )
