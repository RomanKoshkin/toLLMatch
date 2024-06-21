import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'


def get_df_from_log(log_file):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    index_list = []
    prediction_list = []
    char_list = []
    delay_list = []
    elapsed_list = []
    source_length_list = []
    for line in data:
        for char, delay, elapsed in zip(line['prediction'], line['delays'], line['elapsed']):
            index_list.append(line['index'])
            prediction_list.append(line['prediction'])
            char_list.append(char)
            delay_list.append(delay)
            elapsed_list.append(elapsed)
            source_length_list.append(line['source_length'])

    df = pd.DataFrame({
        'index': index_list,
        'prediction': prediction_list,
        'char': char_list,
        'delay': delay_list,
        'elapsed': elapsed_list,
        'source_length': source_length_list
    })
    return df


def parse_log_and_plot(df, index, output_path, chunk_size):
    # Filter the DataFrame for the given index
    df_subset = df[df['index'] == index]

    # Remove consecutive duplicates for 'delay' and 'elapsed'
    unique_delay = df_subset['delay'].loc[df_subset['delay'].shift() != df_subset['delay']]
    unique_elapsed = df_subset['elapsed'].loc[df_subset['elapsed'].shift() != df_subset['elapsed']]

    # Concatenate characters with the same delay
    delay_char_dict = {}
    for char, delay in zip(df_subset['char'], df_subset['delay']):
        if delay in delay_char_dict:
            delay_char_dict[delay] += char
        else:
            delay_char_dict[delay] = char

    # Define source length
    source_length = df_subset["source_length"].iloc[0]

    # Calculate speech chunks
    chunks = np.arange(0, source_length, chunk_size)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot 'speech chunk', 'delay' and 'elapsed' as scatter plots with output timings only
    ax.scatter(chunks, [0]*len(chunks), label="Speech Chunk", marker='|', color='blue')
    ax.scatter(unique_delay, [1]*len(unique_delay), label="Delay (S2T)", marker='|', color='orange')
    ax.scatter(unique_elapsed, [2]*len(unique_elapsed), label="Elapsed (S2T)", color='red', marker='|')

    # Annotate the plot with the characters
    x = 0.2
    for delay, char in delay_char_dict.items():
        ax.annotate(char, (delay, 1+x), fontsize=6, va='center')
        x = x * -1

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Output Timing')
    ax.set_yticks([0, 1, 2])  # Show 0, 1, and 2 on the y-axis
    ax.legend(loc='lower right')

    # Set y-axis limits to make markers appear larger
    ax.set_ylim([-0.5, 2.5])

    # Set x-axis limits to be the same for both subplots
    xmin = 0
    xmax = max(chunks.max(), df_subset['delay'].max(), df_subset['elapsed'].max())
    xmin -= 1000
    xmax += 1000
    ax.set_xlim([xmin, xmax])

    plt.tight_layout()

    # Save the figure to the output path
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a log file and plot the data.')
    parser.add_argument('--log_file', type=str, help='Path to the log file.')
    parser.add_argument('--index', type=int, help='Index of the data to plot.')
    parser.add_argument('--output_path', type=str, help='Path to save the output plot.')
    parser.add_argument('--chunk_size', type=int, default=650, help='Size of the speech chunk in milliseconds.')
    
    args = parser.parse_args()

    df = get_df_from_log(args.log_file)
    parse_log_and_plot(df, args.index, args.output_path, args.chunk_size)

