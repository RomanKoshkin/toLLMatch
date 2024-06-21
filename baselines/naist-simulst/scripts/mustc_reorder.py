import sys
import yaml
from itertools import groupby


def read_and_sort_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    sorted_yaml = []
    for wav_filename, _seg_group in groupby(yaml_data, lambda x: x["wav"]):
        # sort by offset (*string*)
        seg_group = sorted(_seg_group, key=lambda x: str(x["offset"]))
        for segment in seg_group:
            sorted_yaml.append(segment)
    return sorted_yaml

def read_txt_file(txt_file):
    with open(txt_file, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def combine_and_sort(txt_lines, yaml_sorted):
    combined = list(zip(txt_lines, yaml_sorted))

    combined_sorted = []
    for wav_filename, _seg_group in groupby(combined, lambda x: x[1]["wav"]):
        seg_group = sorted(_seg_group, key=lambda x: float(x[1]["offset"]))
        for item in seg_group:
            combined_sorted.append(item)

    return combined_sorted

def save_sorted_txt(combined_sorted, output_file):
    with open(output_file, "w") as file:
        for item in combined_sorted:
            file.write(f"{item}\n")

if __name__ == "__main__":
    yaml_sorted = read_and_sort_yaml(sys.argv[1])
    txt_lines = read_txt_file(sys.argv[2])
    combined_sorted = combine_and_sort(txt_lines, yaml_sorted)
    save_sorted_txt([item[0] for item in combined_sorted], sys.argv[2] + ".ordered")

