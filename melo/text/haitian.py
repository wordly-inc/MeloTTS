import re
from phonemizer import phonemize

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
      - Splits the text (by whitespace) into words.
      - Uses the phonemizer with the eSpeak backend (language "ht") to generate IPA.
      - Distributes the resulting phonemes into a `word2ph` mapping.
      - Optionally pads the start and end of the phoneme list.
    """
    if tokenized is None:
        # For simplicity, assume words are space-delimited.
        tokenized = text.split()
    
    phones = []
    tones = []  # Placeholder for tone information (unused here)
    word2ph = []  # List mapping each word token to its number of phonemes

    for word in tokenized:
        # Use the phonemizer to get an IPA string for the word using eSpeak for Haitian Creole.
        ipa = phonemize(word, language="ht", backend="espeak", strip=True, phoneme_separator=" ")
        # Split the IPA output by spaces into phoneme symbols.
        phone_list = ipa.split()
        if len(phone_list) == 0:
            phone_list = ['UNK']
        phone_len = len(phone_list)
        # For a single word, simply set the mapping to the phone count.
        word2ph.extend(distribute_phone(phone_len, 1))
        phones.extend(phone_list)
        tones.extend([0] * phone_len)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph

if __name__ == "__main__":
    sample_text = "Bonjou, kijan ou ye? Mwen byen, mesi!"
    norm_text = text_normalize(sample_text)
    print("Normalized text:", norm_text)
    
    phones, tones, word2ph = g2p(norm_text)
    print("Phonemes:", phones)
    print("Tones:", tones)
    print("Word2Ph Mapping:", word2ph)
