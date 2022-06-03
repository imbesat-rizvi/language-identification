import re
from string import punctuation


def preprocess_text(text):
    # replace numbers and punctuations with whitespace
    # and convert to lower case
    pattern = f"[{punctuation}0-9]"
    text = re.sub(pattern, " ", text.lower())
    return text