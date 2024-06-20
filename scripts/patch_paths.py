import os

assert os.getcwd().split("/")[-1] == "scripts", "Please run this script from toLLMatch/scripts"
PROJECT_ROOT = "/".join(os.getcwd().split("/")[:-1])
files_to_patch = [
    "ambiguity_en",
    "fleurs_en_de",
    "fleurs_en_de_102",
    "fleurs_en_ru",
    "src_ted_new_tst_100_abspath.de",
    "src_ted_new_tst_100.de",
    "ted_tst_2024",
    "ted_tst_2024_102",
]

for fname in files_to_patch:
    with open(f"{PROJECT_ROOT}/evaluation/SOURCES/{fname}", "r") as f:
        l = f.read()

    with open(f"{PROJECT_ROOT}/evaluation/SOURCES/{fname}", "w") as f:
        f.write(l.replace("PROJECT_ROOT", PROJECT_ROOT))


with open(f"{PROJECT_ROOT}/baselines/FBK-fairseq/RUN_EVAL_ted2024.sh", "r") as f:
    l = f.read()

with open(f"{PROJECT_ROOT}/baselines/FBK-fairseq/RUN_EVAL_ted2024.sh", "w") as f:
    f.write(l.replace("<ABS_PATH_TO_PROJECT_ROOT>", PROJECT_ROOT))


with open(
    f"{PROJECT_ROOT}/baselines/FBK-fairseq/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_edatt.py",
    "r",
) as f:
    l = f.read()

with open(
    f"{PROJECT_ROOT}/baselines/FBK-fairseq/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_edatt.py",
    "w",
) as f:
    f.write(l.replace("PROJECT_ROOT", PROJECT_ROOT))

with open(
    f"{PROJECT_ROOT}/baselines/FBK-fairseq/examples/speech_to_text/simultaneous_translation/agents/v1_1/simul_offline_edatt.py",
    "r",
) as f:
    l = f.read()

with open(
    f"{PROJECT_ROOT}/baselines/FBK-fairseq/examples/speech_to_text/simultaneous_translation/agents/v1_1/simul_offline_edatt.py",
    "w",
) as f:
    f.write(l.replace("PROJECT_ROOT", PROJECT_ROOT))


with open(f"{PROJECT_ROOT}/baselines/seamless-streaming/RUN_ted_tst_2024_ende.sh", "r") as f:
    l = f.read()

with open(f"{PROJECT_ROOT}/baselines/seamless-streaming/RUN_ted_tst_2024_ende.sh", "w") as f:
    f.write(l.replace("<ABS_PATH_TO_PROJECT_ROOT>", PROJECT_ROOT))
