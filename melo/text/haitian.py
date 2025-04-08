import re
from phonemizer import phonemize
from transformers import AutoTokenizer

MODEL_ID = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
def distribute_phone(n_phone, n_word):
    """
    Evenly distribute `n_phone` phonemes across `n_word` parts.
    """
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    """
    Apply basic text normalization for Haitian Creole.
    Adjust or extend this with more specific rules if required.
    """
    text = text.strip()  # remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces to one
    return text

def g2p(text, pad_start_end=True, tokenized=None):
    """
    Converts Haitian Creole text to its phonemic representation.
    This function:
      - Tokenize ht text into tokens.
      - Uses the phonemizer with the eSpeak backend (language "ht") to generate IPA.
      - Distributes the resulting phonemes into a `word2ph` mapping.
      - Optionally pads the start and end of the phoneme list.
    """
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    ph_groups = []
    for t in tokenized:
        # if it's not a subword token append to the group
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", phonemize(w, language="ht", backend="espeak", strip=True)))
    
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    from . import haitian_bert
    return haitian_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    sample_text = "Lavwadlamerik se pi gwo òganizasyon nouvèl medya entènasyonal Lèzetazini."
    norm_text = text_normalize(sample_text)
    print("Normalized text:", norm_text)
    
    phones, tones, word2ph = g2p(norm_text)
    print(word2ph)
    bert = get_bert_feature(norm_text, word2ph)
    print("Phonemes:", phones)
    print(len(phones), tones, sum(word2ph), bert.shape)
