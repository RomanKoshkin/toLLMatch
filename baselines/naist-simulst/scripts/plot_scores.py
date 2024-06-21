import sys
from google.oauth2.service_account import Credentials
import gspread
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import seaborn as sns
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
sns.set_theme(style="white", context="talk")
current_palette = sns.color_palette(n_colors=24)
font = 12
color = f"seaborn"


def main():

    parser = argparse.ArgumentParser(
        description="Plot quality-latency tradeoff curves using the spreadsheet."
    )
    parser.add_argument(
        "--credential",
        type=str,
        required=True,
        help="credential file path for GCP",
    )
    parser.add_argument(
        "--spreadsheet_key",
        type=str,
	default="1C1q7l2_b7oKr8bTWgHkl61_OwCMyl4FJ1XjGw4OwQD8",
        help="Key of the spreadsheet",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="System scores",
        help="sheet's name of the data",
    )
    parser.add_argument(
        "--exp_ids",
        type=int,
        nargs="+",
        required=True,
        help="list of exp_id to plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="test",
        help="title of the plot",
    )
    parser.add_argument(
        "--set_title",
        action="store_true",
    )
    parser.add_argument(
        "--legend_fontsize",
        type=str,
        default="medium",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="save directory path"
    )
    parser.add_argument(
        "--colors",
        type=str,
        nargs="+",
        default=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        help="color settings",
    )
    parser.add_argument(
        "--marks",
        type=str,
        nargs="+",
        default=[".", "s", "p", "h", "x", "*", "+", "d", "^", "2"],
        help="mark settings",
    )
    parser.add_argument(
        "--linestyles",
        type=str,
        nargs="+",
        default=["solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid"],
        help="linestyle settings",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--no_label",
        action="store_true",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="tst-COMMON_v2",
        help="evaluation data",
    )
    parser.add_argument(
        "--x_metric",
        type=str,
        default="AL",
    )
    parser.add_argument(
        "--y_metric",
        type=str,
        default="BLEU",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="xlim settings",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[-1, -1],
        help="ylim settings",
    )
    parser.add_argument(
        "--data_label",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
    )

    args = parser.parse_args()

    # Load data frame from spreadsheet
    MY_CREDENTIAL_FILEPATH = args.credential
    SPREADSHEET_KEY = args.spreadsheet_key
    SHEET = args.sheet
    
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    credentials = Credentials.from_service_account_file(
        MY_CREDENTIAL_FILEPATH,
        scopes=scopes
    )
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key(SPREADSHEET_KEY).worksheet(SHEET)
    df = pd.DataFrame(worksheet.get_all_records())
    
    # Set plt configurations
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['font.size'] = font
    plt.rcParams['mathtext.fontset'] = 'stix'
    ratio = 1.618
    plt.rcParams['figure.figsize'] = (5 * ratio, 5)
    plt.rcParams["legend.framealpha"] = 0.8
    
    legend_colors = defaultdict(str)
    legend_marks = defaultdict(str)
    
    assert len(args.exp_ids) <= len(args.colors)
    assert len(args.exp_ids) <= len(args.marks)
    
    grouped = df.groupby("exp_id")

    fig, ax = plt.subplots()
    for exp_id, group in grouped:
        if not exp_id in args.exp_ids:
            continue
        print(f"exp_id = {exp_id}")
        x_scores = []
        y_scores = []
        data_labels = []
        highlight_x_scores = []
        highlight_y_scores = []
        label = None
        if args.labels != [] and not args.no_label:
            label = args.labels.pop(0)
        for i, row in group.iterrows():
            if not label and not args.no_label:
                label = row["Model"]
            if row[args.x_metric] == ""  or row[args.y_metric] == "":
                continue
            if row["Eval_data"] != args.eval_data:
                continue
            if row["Ignore"] == "TRUE":
                continue
            x_scores.append(float(row[args.x_metric]))
            y_scores.append(float(row[args.y_metric]))
            if row["Highlight"] == "TRUE":
                highlight_x_scores.append(x_scores[-1])
                highlight_y_scores.append(y_scores[-1])
            if args.data_label:
                data_labels.append(row[args.data_label])
            else:
                data_labels.append(0)

        color = args.colors.pop(0)
        marker = args.marks.pop(0)
        linestyle = args.linestyles.pop(0)
        sorted_pairs = sorted(zip(x_scores, y_scores, data_labels))
        x_scores, y_scores, data_labels = zip(*sorted_pairs)
        x_scores, y_scores, data_labels = list(x_scores), list(y_scores), list(data_labels)

        if args.xlim[0] != -1:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        if args.ylim[0] != -1:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.plot(
            x_scores, y_scores, label=label, marker=marker,
            linestyle=linestyle,
            color=color, markersize=6, alpha=1.0,
        )
        if args.data_label:
            for i, data_label in enumerate(data_labels):
                ax.text(
                    x_scores[i]-20, y_scores[i]+0.1,
                    data_label, color=color, fontsize="large"
                )
        if args.highlight and len(highlight_x_scores) > 0:
            sorted_pairs = sorted(zip(highlight_x_scores, highlight_y_scores))
            x_scores, y_scores = zip(*sorted_pairs)
            x_scores, y_scores = list(x_scores), list(y_scores)
            ax.plot(
                x_scores, y_scores, marker="o", markersize=12,
                markerfacecolor='None', markeredgecolor=color,
                markeredgewidth=1.5,
                alpha=1.0, linestyle="None",
            )

    ax.set_xlabel(args.x_metric)
    ax.set_ylabel(args.y_metric)
    ax.grid()
    if args.set_title:
        ax.set_title(args.title)
    if not args.no_label:
        plt.legend(fontsize=args.legend_fontsize)
    save_path = Path(args.save_dir) / Path(args.title + ".pdf")
    plt.savefig(save_path, transparent=False, bbox_inches='tight')
    save_path = Path(args.save_dir) / Path(args.title + ".png")
    plt.savefig(save_path, transparent=False, bbox_inches='tight')

if __name__ == "__main__":
    main()
