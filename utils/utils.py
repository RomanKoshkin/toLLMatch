# from numba import njit
import numpy as np
import time

# from numba.typed import List
from typing import List
import os
import shutil


def recov(it):
    if it[0] == "_":
        return it[1]
    else:
        return it[0]
    
# --- begin old version ---
# @njit
def count_matches(i: str, j: str):
    m = 0
    s = ""
    for aa, jj in zip(i, j):
        if (aa == "_") or (jj == "_"):
            continue
        if aa == jj:
            m += 1
            s += aa
    return m, s


class Timer:

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f"{self.interval:.4f} s.")
        if type is KeyboardInterrupt:
            print("KeyboardInterrupt caught. Cleaning up...")
            return True  # This will suppress the exception


def pad(_a, _b, overlap):
    a = _a + ["_"] * (len(_b) - overlap)
    b = ["_"] * (len(_a) - overlap) + _b
    return a, b


def update_source_word_list(a: List[str], b: List[str], verbose=False) -> List[str]:
    """
    Update the source word list with NEW words from list b.
    Assumes that b extends a.

    Args:
        a (List[str]): The original word list.
        b (List[str]): The updated word list.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Tuple[List[str], int]: A tuple containing the updated word list and the number of words added.
    """
    old_len = len(a)
    A = []
    for overlap in range(len(a) + 1):
        r = [(i, j, i == j) for i, j in zip(*pad(a, b, overlap))]
        A.append(sum([t[-1] for t in r]))
    res = [(i, j, i.lower() == j.lower()) for i, j in zip(*pad(a, b, np.argmax(A)))]
    new_a = list(map(recov, res))
    new_a = remove_adjacent_duplicates(new_a)
    num_words_added = len(new_a) - old_len
    return new_a, num_words_added


# --- end new version ---


def purge_directory(directory_path):
    """
    Recursively deletes all files and directories within the specified directory.

    Args:
        directory_path (str): The path of the directory to be purged.

    Raises:
        None

    Returns:
        None
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Iterate over each item in the directory
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            try:
                # If item is a file or a symlink, delete it
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                # If item is a directory, delete it and all its contents
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
    else:
        print(f"Directory {directory_path} does not exist.")


def remove_adjacent_duplicates(strings):
    """
    Removes adjacent duplicates from a list of strings.

    Args:
        strings (list): A list of strings.

    Returns:
        list: A new list with adjacent duplicates removed.

    Example:
        >>> remove_adjacent_duplicates(['apple', 'banana', 'banana', 'orange', 'orange', 'orange'])
        ['apple', 'banana', 'orange']
    """
    if not strings:
        return []
    result = [strings[0]]  # Initialize the result list with the first element
    for i in range(1, len(strings)):  # Iterate from the second element to the end
        # Check if the current string is different from the last string in the result
        if strings[i] != result[-1]:
            result.append(strings[i])
    return result


def parse_language_pair(pair: str):
    a, b = pair.split("-")

    if a == "en":
        SRC_LANG = "English"
    elif a == "de":
        SRC_LANG = "German"
    elif a == "ru":
        SRC_LANG = "Russian"
    elif a == "it":
        SRC_LANG = "Italian"
    elif a == "es":
        SRC_LANG = "Spanish"
    elif a == "fr":
        SRC_LANG = "French"
    else:
        raise RuntimeError("Unknown source langugage")

    if b == "en":
        TGT_LANG = "English"
    elif b == "de":
        TGT_LANG = "German"
    elif b == "ru":
        TGT_LANG = "Russian"
    elif b == "it":
        TGT_LANG = "Italian"
    elif b == "es":
        TGT_LANG = "Spanish"
    elif b == "fr":
        TGT_LANG = "French"
    else:
        raise RuntimeError("Unknown target langugage")

    return SRC_LANG, TGT_LANG
