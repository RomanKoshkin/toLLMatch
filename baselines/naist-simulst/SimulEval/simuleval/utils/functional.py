# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import closing
import socket
from typing import List
import soundfile as sf
import numpy as np

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def concat_itts_audios(
    audio_path_list:List[str],
    chunk_timestamp_list:List[List[float]],
    chunk_duration_list: List[List[float]],
    mergin:float=100,
) -> np.ndarray:
    
    assert len(audio_path_list) == len(chunk_timestamp_list)
    assert len(audio_path_list) == len(chunk_duration_list)
    
    # read audio data
    audio_list = []
    sampling_rate = -1
    for audio_path in audio_path_list:
        audio, sampling_rate = sf.read(audio_path)
        audio_list.append(audio)
    
    def convert_ms_to_sample(ms_duration: float):
        return int(sampling_rate * ms_duration / 1000)
    
    # save offset from timestamps
    sampled_chunk_offset_list = []
    for chunk_timestamp in chunk_timestamp_list: # list in list[list]
        sampled_chunk_offsets = []
        for offset in chunk_timestamp: # float in list
            sampled_chunk_offsets.append(
                convert_ms_to_sample(offset - chunk_timestamp[0])
            )
        sampled_chunk_offset_list.append(sampled_chunk_offsets)

    # duration for splitting audio by chunk
    sampled_chunk_duration_list = []
    for chunk_durations in chunk_duration_list:
        sampled_chunk_duration = [convert_ms_to_sample(duration) for duration in chunk_durations]
        sampled_chunk_duration_list.append(sampled_chunk_duration)
    
    # offset for splitting audio by segment
    sampled_segment_offset_list = [convert_ms_to_sample(timestamps[0]) for timestamps in chunk_timestamp_list]
    # mergins between segments
    segment_mergin_length = convert_ms_to_sample(mergin)
    
    
    for i, (audio, sampled_chunk_offsets, sampled_chunk_durations) in enumerate(zip(
        audio_list, sampled_chunk_offset_list, sampled_chunk_duration_list,
    )):
        concat_audio_list = []
        total_length = 0
        processed_duration = 0
        
        for chunk_offset, chunk_duration in zip(sampled_chunk_offsets, sampled_chunk_durations):
            if total_length < chunk_offset:
                padding_length = chunk_offset - total_length
                zero_audio = np.zeros(padding_length)
                
                total_length += padding_length
                concat_audio_list.append(zero_audio)
            
            concat_audio_list.append(audio[processed_duration:processed_duration+chunk_duration])
            total_length += chunk_duration
            processed_duration += chunk_duration
            
        audio_list[i] = np.concatenate(concat_audio_list)
    
    
    concat_audio_list = []
    total_length = 0
    
    for audio, offset in zip(audio_list, sampled_segment_offset_list):
        
        if total_length < offset:
            padding_length = offset - total_length
            zero_audio = np.zeros(padding_length)
            
            total_length += padding_length
            concat_audio_list.append(zero_audio)
        elif total_length > offset:
            zero_audio = np.zeros(segment_mergin_length)
            
            total_length += segment_mergin_length
            concat_audio_list.append(zero_audio)
        
        total_length += audio.shape[0]
        concat_audio_list.append(audio)
    
    concat_audio = np.concatenate(concat_audio_list)
    return concat_audio