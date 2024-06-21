import re
import nltk
from sacremoses import MosesPunctNormalizer
import html

nltk.download("punkt")
mpn = MosesPunctNormalizer()


def clean_speaker_name(text: str) -> str:
    """
    Checks if the text starts with the pattern: [speaker name followed
    by colon] and returns a clean version of it by removing the pattern
    """

    if ": " in text:
        for sentence in nltk.sent_tokenize(text):
            if ": " in sentence:
                start_text, rest_text = sentence.split(": ", maxsplit=1)
                start_tokens = re.sub(" +", " ", start_text).strip().split(" ")
                num_start_tokens = len(start_tokens)

                # XXX: one word, initials, all caps
                if num_start_tokens == 1 and start_tokens[0].isupper():
                    text = text.replace(sentence, rest_text)
                # Xxxx (Zzzz) Yyyy: two or three words, first (middle) last, start of each name is capitalized
                elif 1 < num_start_tokens < 4 and all(
                    [start_tokens[i][0].isupper() for i in range(num_start_tokens)]
                ):
                    text = text.replace(sentence, rest_text)

    return text


def clean_event(text: str) -> str:
    
    # just parenthesis
    simple_event_pattern = r"\([^()]*\)"

    if ": " in text:
        for event in re.findall(simple_event_pattern, text):

            # check if event contains actual text from a speaker: (XX: utterance) -> utterance
            if ": " in event:
                event_text = event[1:-1]  # (xyz) -> xyz
                event_text_cleaned = clean_speaker_name(event_text)

                # replace event with its cleaned text
                if event_text != event_text_cleaned:
                    text = text.replace(event, event_text_cleaned)

    # remove rest of the events
    # parenthesis with punctuations, " . ... :, before or after 
    all_event_patterns = r'"(\([^()]*\))"|"(\([^()]*\))|(\([^()]*\):)|(\([^()]*\)\.\.\.)|(\([^()]*\)\.)|(\([^()]*\))'
    text = re.sub(all_event_patterns, "", text)
    
    text = text.replace(" -- -- ", " -- ")

    return text


def general_utterance_cleaner(text: str) -> str:

    text = html.unescape(bytes(text, "utf-8").decode("utf-8", "ignore"))

    # tabs, newlines and trailing spaces
    text = re.sub(" +", " ", text.strip().replace("\t", " ").replace("\n", " "))

    # normalizes punctuation
    text = mpn.normalize(text)
    
    # normalization bug
    text = text.replace(',"', '",')
    text = text.replace(",'", "',")

    # empty if it does not contain at least two consequitve letters (shortest word)
    if len(text) and max([len(t) for t in text.split()]) < 2:
        text = ""

    return text


def mustc_utterance_cleaner(text: str) -> str:

    # remove events (eg: LACHEN)
    text = clean_event(text)

    # remove speaker name and colon
    text = clean_speaker_name(text)

    text = general_utterance_cleaner(text)

    return text


def europarlst_utterance_cleaner(text: str) -> str:

    # Fixes spaces in numbers > 1000 with inclusing a comma (50 000 -> 50,000)
    # For consitency with MuST-C data
    search = True
    while search:
        num = re.search(r"\d\s\d{3}", text)
        if num:
            text = text.replace(num[0], num[0].replace(" ", ","))
        else:
            search = False

    text = general_utterance_cleaner(text)
    
    # extra space between quotes
    text = text.replace(" ' s ", "'s ")
    for pattern in re.findall(f"(' [^']+ ')", text) + re.findall(f'(" [^"]+ ")', text):
        text = text.replace(pattern, pattern[0] + pattern[2:-2] + pattern[-1])
    text = text.replace("s ' ", "s' ")
    
    return text


def covost_utterance_cleaner(text: str) -> str:

    # very frequent double quotes in covost
    text = text.replace('""', '"')
    text = general_utterance_cleaner(text)
    return text
