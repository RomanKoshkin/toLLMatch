import sys
from pathlib import Path
import json
import yaml
import itertools
from simuleval.data import SpeechSegmentLoader

wav_dirpath = sys.argv[1]
yaml_filepath = sys.argv[2]
log_filepath = Path(sys.argv[3])

# Load instances.log
log_data = []
with open(log_filepath) as f:
    for line in f.readlines():
        log_data.append(json.loads(line))
        log_data[-1]["source"] = [log_data[-1]["source"]] \
            if not isinstance(log_data[-1]["source"], list) \
            else log_data[-1]["source"]
        log_data[-1]["orig_wav_filename"] = \
            "_".join(
                Path(log_data[-1]["source"][0]).stem.split("_")[:-1]
            ) + ".wav"

# Add segment["timestamp"] to JSON data
for wav_filename, talk_group in itertools.groupby(
    log_data, key=lambda x: x["orig_wav_filename"]
):
    wav_filepath = Path(wav_dirpath) / wav_filename
    loader = SpeechSegmentLoader(
        str(wav_filepath), yaml_filepath
    )
    for segment in talk_group:
        segment_offset = next(loader)[1] * 1000

        segment["timestamp"] = [
            segment_offset + delay
            for delay in segment["delays"]
        ]
        segment["segment_offset"] = segment_offset

# Write instances_timestamp.log
with open(
    log_filepath.parent / "instances_timestamp.log", "w"
) as f:
    for segment in log_data:
        f.write(json.dumps(segment, ensure_ascii=False) + "\n")
