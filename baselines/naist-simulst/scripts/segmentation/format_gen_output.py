import argparse


def format_gen_output(path_to_input: str, path_to_output: str) -> None:

    # parse the txt with the generation output from fairseq-generate
    # generated sentences start with D-$i where $i is the correct order
    # of the sentence in the dataset
    raw_generation, correct_order = [], []
    with open(path_to_input, "r", encoding = "utf8") as f:
        for line in f.read().splitlines():
            if line[:2] == "D-":
                correct_order.append(int(line.split(maxsplit = 1)[0].split("D-")[-1]))
                splits = line.split(maxsplit = 2)
                if len(splits) == 3:
                    raw_generation.append(splits[2])
                else:
                    raw_generation.append("")

    # fix to correct order
    raw_generation = [gen for _, gen in sorted(zip(correct_order, raw_generation))]

    # save clean generation txt file
    with open(path_to_output, "w", encoding = "utf8") as f:
        for line in raw_generation:
            f.write(line + "\n")

    print(f"Saved formatted generation at {path_to_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_input", "-i", required=True, type=str)
    parser.add_argument("--path_to_output", "-o", required=True, type=str)
    args = parser.parse_args()

    format_gen_output(args.path_to_input, args.path_to_output)