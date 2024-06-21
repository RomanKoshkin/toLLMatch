import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-filepath",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    
    manifest_file = open(args.manifest_filepath, mode="r", encoding="utf-8")
    
    ratio_list = []
    for i, manifest_str in enumerate(manifest_file):
        # skip header
        if i == 0: continue
        
        manifest = manifest_str.split("\t")
        n_samples = int(manifest[2])
        n_tokens = len(manifest[3].split(" "))
        ratio_list.append(n_samples / n_tokens)
    
    ratio_average = sum(ratio_list) / len(ratio_list)
    
    print("ratio average:", ratio_average)    



if __name__=="__main__":
    main()