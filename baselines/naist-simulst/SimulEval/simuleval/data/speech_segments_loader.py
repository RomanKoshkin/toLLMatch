import soundfile as sf
import yaml
from typing import Tuple
import numpy as np

class SpeechSegmentLoader:
    def __init__(self, wav_filepath: str, yaml_filepath: str) -> None:
        """The dataloader for speech data with segmentation

        Args:
            wav_filepath (str): full path to wav file (.wav)
            yaml_filepath (str): full path to yaml file (.yaml)
        """
        wav_filename = wav_filepath.split("/")[-1]
        self.yaml_filepath = yaml_filepath
        self.wav_data, self.sampling_rate = sf.read(wav_filepath)
        
        with open(yaml_filepath, mode="r", encoding="utf-8") as yaml_file:
            full_manifests = yaml.safe_load(yaml_file)
            self.manifests = []
            for manifest in full_manifests:
                if manifest["wav"] == wav_filename:
                    self.manifests.append(manifest)
        
        # variable for iteration
        self.current_iter = 0
    
    
    def __len__(self) -> int:
        return len(self.manifests)
        
        
    def __iter__(self):
        return self
    
    
    def __next__(self) -> Tuple[np.ndarray, float]:
        if self.current_iter == len(self.manifests):
            raise StopIteration()
        
        current_manifest = self.manifests[self.current_iter]
        
        start_sec = current_manifest["offset"]
        end_sec = current_manifest["offset"] + current_manifest["duration"]
        
        start_pos = int(self.sampling_rate * start_sec)
        end_pos = int(self.sampling_rate * end_sec)
        
        speech_segment = self.wav_data[start_pos:end_pos]
        
        self.current_iter += 1
        return speech_segment, start_sec