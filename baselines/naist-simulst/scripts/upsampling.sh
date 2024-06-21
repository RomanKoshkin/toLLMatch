#!/bin/bash

# コマンドライン引数をチェック
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <file_a> <file_b> <output_file>"
    exit 1
fi

# ファイル名を変数に代入
file_a=$1
file_b=$2
output_file=$3

# ファイルaとbの行数を取得
N=$(wc -l < "$file_a")
M=$(wc -l < "$file_b")

# bのアップサンプリング係数を計算
upsampling_factor=$(awk -v n="$((N-1))" -v m="$((M-1))" 'BEGIN { print int(n / m) }')

# ファイルbの2行目以降をアップサンプリングし、一時ファイルに保存
awk -v factor="$upsampling_factor" '{
    if (NR > 1) {
        for (i = 1; i <= factor; i++) {
            print $0
        }
    }
}' "$file_b" > b_upsampled.txt

# b_upsampled.txtの行数を取得
M_upsampled=$(wc -l < b_upsampled.txt)

# a.txtの1行目を出力ファイルに書き込む
head -n 1 "$file_a" > "$output_file"

# ファイルaの2行目以降とアップサンプリングされたファイルbを交互に結合して、出力ファイルに追加
paste -d'\n' <(head -n "$((M_upsampled+1))" "$file_a" | tail -n +2) b_upsampled.txt >> "$output_file"

# a.txtの未使用行を出力ファイルの末尾に追加
tail -n +"$((M_upsampled+2))" "$file_a" >> "$output_file"

# 一時ファイルを削除
rm b_upsampled.txt
