from simuleval.utils.functional import concat_itts_audios

from argparse import ArgumentParser
import json
from tqdm import tqdm
import soundfile as sf
import os
from typing import Dict

def main():
    parser = ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()
    
    
    prediction_path_dict: Dict[str, list] = {} # orig_wav_filename -> list of prediction (path)
    duration_dict: Dict[str, list] = {}
    timestamp_dict: Dict[str, list] = {}
    
    with open(args.log_path, mode="r", encoding="utf-8") as log_file:
        for log_line in log_file:
            log_dict = json.loads(log_line)
            
            wav_filename = log_dict["orig_wav_filename"]
            prediction_path = log_dict["prediction"]
            durations = log_dict["durations"]
            timestamps = log_dict["timestamp"]
            
            if prediction_path_dict.get(wav_filename, None) is None:
                prediction_path_dict[wav_filename] = []
                duration_dict[wav_filename] = []
                timestamp_dict[wav_filename] = []
            
            prediction_path_dict[wav_filename].append(prediction_path)
            duration_dict[wav_filename].append(durations)
            timestamp_dict[wav_filename].append(timestamps)
    
    for wav_name in tqdm(prediction_path_dict.keys(), desc="TTS data is being concated"):
        concat_audio = concat_itts_audios(
            audio_path_list=prediction_path_dict[wav_name],
            chunk_timestamp_list=timestamp_dict[wav_name],
            chunk_duration_list=duration_dict[wav_name],
            mergin=100,
        )
        
        output_filepath = os.path.join(args.output_dir, f"pred_{wav_name}")
        
        sf.write(
            file=output_filepath,
            data=concat_audio,
            samplerate=args.sampling_rate,
        )


if __name__=="__main__":
    main()