import ast
import sys
from pathlib import Path

def main():
    log_path = Path(sys.argv[1])
    stack = []
    for result in log_path.open().readlines():
        result = result.rstrip("\r\n")
        result = ast.literal_eval(result)
        stack.append(result["prediction"]+"\n")
    (log_path.parent/"generation.txt").open("w").writelines(stack)

if __name__ == "__main__":
    main()
