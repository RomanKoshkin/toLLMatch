import argparse
import torch
import librosa
import glob
import numpy as np
from tqdm import tqdm

def evaluate_utmos(audio_dir, ignore_files, sampling_rate, use_gpu=True):
    # UTMOS predictorを読み込む
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    if use_gpu:
        predictor.cuda()

    # 除外するファイルのリストを準備
    ignore_list = ignore_files.split(',')

    # 指定されたディレクトリ内の全WAVファイルを検索
    audio_files = glob.glob(f"{audio_dir}/*.wav")
    # 除外ファイルをフィルタリング
    audio_files = [file for file in audio_files if not any(ignore_file in file for ignore_file in ignore_list)]
    
    scores = []

    # 各ファイルのUTMOSスコアを計算
    for audio_path in tqdm(audio_files):
        wave, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
        if use_gpu:
            score = predictor(torch.from_numpy(wave).unsqueeze(0).cuda(), sr)
        else:
            score = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
        scores.append(score.item())

    # スコアの平均を計算して表示
    if scores:
        average_score = np.mean(scores)
        print(f"Average UTMOS score = {average_score}")
    else:
        print("No audio files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the naturalness of audio files in a directory using UTMOS.')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files to evaluate.')
    parser.add_argument('--ignore_files', type=str, default="", help='Comma-separated list of audio files to ignore.')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of audio files.')

    args = parser.parse_args()
    evaluate_utmos(args.audio_dir, args.ignore_files, args.sampling_rate)
