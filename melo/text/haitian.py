import re
from phonemizer.backend import ESpeakBackend  # Import the backend class directly
from transformers import AutoTokenizer

MODEL_ID = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Initialize the eSpeak backend for Haitian Creole.
phoneme_backend = ESpeakBackend("ht", strip=True)

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

def number_to_haitian_creole(num):
    units = {0: "zewo", 1: "youn", 2: "de", 3: "twa", 4: "kat", 5: "senk",
             6: "sis", 7: "sèt", 8: "uit", 9: "nèf"}
    teens = {10: "dis", 11: "onz", 12: "douz", 13: "trèz", 14: "katorz",
             15: "kenz", 16: "sèz", 17: "disèt", 18: "dizuit", 19: "diznèf"}
    tens = {20: "ven", 30: "trant", 40: "karant", 50: "senkant",
            60: "swasant", 70: "swasant-diz", 80: "katreven", 90: "katreven-diz"}
    num = int(num)
    if num < 10:
        return units[num]
    elif num < 20:
        return teens[num]
    elif num < 100:
        div, mod = divmod(num, 10)
        if mod == 0:
            return tens[div * 10]
        return f"{tens[div * 10]}-{units[mod]}"
    elif num < 1000:
        div, mod = divmod(num, 100)
        if mod == 0:
            return f"{units[div]} san"
        return f"{units[div]} san {number_to_haitian_creole(mod)}"
    elif num < 1000000:
        div, mod = divmod(num, 1000)
        if mod == 0:
            return f"{number_to_haitian_creole(div)} mil"
        return f"{number_to_haitian_creole(div)} mil {number_to_haitian_creole(mod)}"
    else:
        return str(num)

def text_normalize(text):
    """
    Apply basic text normalization for Haitian Creole.
    Adjust or extend this with more specific rules if required.
    """
    text = text.strip()  # remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces to one
    def replace_number(match):
        return number_to_haitian_creole(match.group())
    return re.sub(r'\d+', replace_number, text)

def g2p(text, pad_start_end=True, tokenized=None):
    """
    Converts Haitian Creole text to its phonemic representation.
    This function:
      - Tokenizes text into tokens.
      - Uses the explicitly initialized phonemizer backend with eSpeak (language "ht") to generate IPA.
      - Distributes the resulting phonemes into a `word2ph` mapping.
      - Optionally pads the start and end of the phoneme list.
    """
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    ph_groups = []
    for t in tokenized:
        # If the token does not start with the subword marker, begin a new group.
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            # Remove the subword marker and append.
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
            # Use the already initialized backend's phonemize() method.
            phone_list = list(filter(lambda p: p != " ", phoneme_backend.phonemize(w)))

        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1

        # Evenly distribute the phonemes across subword tokens.
        distribution = distribute_phone(phone_len, word_len)
        word2ph += distribution

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
    print("Word to phoneme mapping:", word2ph)
    
    bert = get_bert_feature(norm_text, word2ph)
    print("Phonemes:", phones)
    print("Details:", len(phones), tones, sum(word2ph), bert.shape)
