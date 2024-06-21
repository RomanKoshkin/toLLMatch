import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))
from demosystem_itts_accent_2.sub2yomi import *
from demosystem_itts_accent_2.base_acct import *
from demosystem_itts_accent_2.pwg import *
from demosystem_itts_accent_2.LabelMake import LabelMake
import torch
import numpy as np
from scipy.io import wavfile
from argparse import ArgumentParser
import pdb
from scipy.io.wavfile import write

try:
    from simuleval.utils import entrypoint
    from simuleval.agents import TextToSpeechAgent, AgentStates
    from simuleval.agents.actions import Action, ReadAction, WriteAction
    from simuleval.data.segments import Segment, TextSegment, EmptySegment, SpeechSegment
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

def remove_strings(text, exclude_list):
    for exclude_str in exclude_list:
        text = text.replace(exclude_str, "")
    return text

class TTSAgentStates(AgentStates):
    def __init__(self,sub2yomi_model_path:str,yomi2tts_model_path:str,tts2wav_model_path:str,\
                sub2yomi_dict_path:str,yomi2tts_phoneme_dict_path:str,yomi2tts_a1_dict_path:str,\
                yomi2tts_a2_dict_path:str,yomi2tts_a3_dict_path:str,yomi2tts_f1_dict_path:str,\
                yomi2tts_f2_dict_path:str) -> None:
        self.devices="cuda" if torch.cuda.is_available() else "cpu" #cpu or gpu
        self.vocabs=sub2yomi.load_vocab(sub2yomi_dict_path) #vocabs for subword2yomi
        self.inv_vocabs=[{i:e for i,e in enumerate(self.vocabs[idx])} for idx in range(len(self.vocabs))] #inversed vocabs
        self.yomi_model=sub2yomi.model_load(sub2yomi.hp_sub2yomi,sub2yomi_model_path,self.vocabs) #suword2yomi model
        self.model1=sub2yomi.sub2yomi(self.yomi_model,self.devices,self.vocabs) #class of subword2yomi model
        self.model2=tts.tts(self.devices,tts.taco2load(yomi2tts_model_path),dict_path=[yomi2tts_phoneme_dict_path,\
                            yomi2tts_a1_dict_path,yomi2tts_a2_dict_path,yomi2tts_a3_dict_path,yomi2tts_f1_dict_path,\
                            yomi2tts_f2_dict_path],wait=3,eos_seq=None) #load trained tacotron2 and class of yomi2feats model
        self.vocoder=pwg.builde_model() #model of feats2speech(vocoder)
        self.vocoder=pwg.load_model(self.vocoder,tts2wav_model_path) #load vocoder
        self.model3=pwg.PWG(self.devices,self.vocoder) #model of vocoder

        #add yanagita
        self.label_make=LabelMake()

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
        self.wait1_idx = 3  # wait count for model1

        self.model2.initialize()
        self.is_first = True  # flag for first input
        self.is_decode_first = True  # flag for first decode
        self.prev_memory2 = None
        # initial mel_inputs for model2
        self.mel_inputs = \
            torch.zeros((1,hp_taco2.num_mels,1),dtype=torch.float).to(self.devices)
        self.out_pos = []
        self.cat_inputs = None  # encoder input of model2

        # Yanagita added.
        # self.tts_input_minimal_length=3
        self.tts_input_chunks=[]

        #Yanagita add.
        self.label_make.initialize()
        self.yomi=""
        self.bound_queue=[]
        self.type_queue=[]
        self.is_pause_lab=False
        self.sos_flag=True
        self.mos_flag=False
        self.wav_out=[] 
        self.filename=0
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


def reduce_commas(arr,arr2,arr3):
    new_arr = []
    new_arr2 = [] 
    new_arr3 = []
    for i, (item,item2,item3) in enumerate(zip(arr,arr2,arr3)):
        if item == "、" and i > 0 and arr[i-1] == "、":
            continue
        new_arr.append(item)
        new_arr2.append(item2)
        new_arr3.append(item3)
    while new_arr and new_arr[-1] == "、":
        new_arr.pop()
        new_arr2.pop()
        new_arr3.pop()
    return new_arr,new_arr2,new_arr3



class Subword2Yomi2SpeechAgent(TextToSpeechAgent):

    def __init__(self, args):
        super().__init__(args)

        self.args = args

    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--tts_input_minimal_length",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--print_tts_input",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--sub2yomi_model_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_model_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--tts2wav_model_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--sub2yomi_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_phoneme_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_a1_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_a2_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_a3_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_f1_dict_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--yomi2tts_f2_dict_path",
            type=str,
            default=None,
        )

    def build_states(self):
        return TTSAgentStates(\
                self.args.sub2yomi_model_path,\
                self.args.yomi2tts_model_path,\
                self.args.tts2wav_model_path,\
                self.args.sub2yomi_dict_path,\
                self.args.yomi2tts_phoneme_dict_path,\
                self.args.yomi2tts_a1_dict_path,\
                self.args.yomi2tts_a2_dict_path,\
                self.args.yomi2tts_a3_dict_path,\
                self.args.yomi2tts_f1_dict_path,\
                self.args.yomi2tts_f2_dict_path\
                )

    def policy(self) -> Action:
        if len(self.states.source) == 0:
            if self.states.source_finished:
                return WriteAction(
                    SpeechSegment(
                        content=[0.0] * 1000,
                        sample_rate=22_050,
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
                    yomi_queue = [
                        self.states.inv_vocabs[1][out[0].data.item()] \
                        if out.data.item() != self.states.model1.yomi_unk_id \
                        else "" for out in outs[0]
                    ]
                    type_queue =[
                        self.states.inv_vocabs[5][out[0].data.item()] \
                        if out.data.item() != self.states.vocabs[5]["<unk>"] \
                        else "" for out in outs[4]]
                    bound_queue =[
                        self.states.inv_vocabs[6][out[0].data.item()] \
                        if out.data.item() != self.states.vocabs[6]["<unk>"] \
                        else "" for out in outs[5]]
                    #print("".join(yomi_queue))
                    for t_yomi,t_type,t_bound in zip(yomi_queue,type_queue,bound_queue):
                      if t_bound!="<ww>":
                        self.states.yomi+=t_yomi
                        self.states.yomi_queue.append(self.states.yomi)
                        self.states.yomi=""
                        self.states.type_queue.append(t_type if t_type not in ["<ww>","<s>","</s>",""] else '0') #ad. hoc for estimating error
                        self.states.bound_queue.append(t_bound)
                      else:
                        self.states.yomi+=t_yomi
            #print(" ".join(self.states.yomi_queue))
            assert len(self.states.yomi_queue)==len(self.states.type_queue) and len(self.states.type_queue)==len(self.states.bound_queue)
            if len(self.states.yomi_queue)==0:
              return ReadAction()
            else:
              self.states.yomi_queue,self.states.bound_queue,self.states.type_queue = reduce_commas(self.states.yomi_queue,self.states.bound_queue,self.states.type_queue)
              label=self.states.label_make.input2label(self.states.yomi_queue,self.states.bound_queue,self.states.type_queue)
              self.states.yomi_queue=[]
              self.states.bound_queue=[]
              self.states.type_queue=[]
              if len(label)==0:
                  return ReadAction()
            self.states.wav_out=[]
            for lab in label:
              if len(lab)==1 and lab[0][0]=="pau":
                self.states.is_pause_lab=True
                continue
              elif self.states.is_pause_lab:
                self.states.is_pause_lab=False
                lab=[self.states.label_make.pau_label]+lab
              phoneme,a1,a2,a3,f1,f2=self.states.label_make.analyze_labels(lab)
              input_phoneme=torch.from_numpy(np.array(self.states.model2.text_to_sequence(phoneme))).unsqueeze(0).long().to(self.states.devices)
              a1=torch.from_numpy(np.array(self.states.model2.a1_to_sequence(a1))).unsqueeze(0).long().to(self.states.devices)
              a2=torch.from_numpy(np.array(self.states.model2.a2_to_sequence(a2))).unsqueeze(0).long().to(self.states.devices)
              a3=torch.from_numpy(np.array(self.states.model2.a3_to_sequence(a3))).unsqueeze(0).long().to(self.states.devices)
              f1=torch.from_numpy(np.array(self.states.model2.f1_to_sequence(f1))).unsqueeze(0).long().to(self.states.devices)
              f2=torch.from_numpy(np.array(self.states.model2.f2_to_sequence(f2))).unsqueeze(0).long().to(self.states.devices)
              model2_memory=self.states.model2.encode(input_phoneme,a1,a2,a3,f1,f2)
              _,post_mel_out,attn,_=self.states.model2.decode(model2_memory,self.states.mel_inputs)
              self.states.mel_inputs=post_mel_out[:,:,-1].unsqueeze(2)
              wav=self.states.model3.synthesis(post_mel_out)
              wav = wav.astype(np.float32) / 32768  # ad-hoc
              max_length = 30  # in seconds
              wav = wav[:max_length*22_050]
              self.states.wav_out.append(wav)            
            if len(self.states.wav_out)!=0:
              wav=np.hstack(self.states.wav_out)
              return WriteAction(
                SpeechSegment(
                    content=wav.tolist(),
                    sample_rate=22_050,
                    finished=self.states.source_finished,
                ),
                finished=self.states.source_finished,
              )
            else:
              return WriteAction(
                SpeechSegment(
                    content=[],
                    sample_rate=22_050,
                    finished=self.states.source_finished,
                ),
                finished=self.states.source_finished,
              )
