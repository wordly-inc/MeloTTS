import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Use a multilingual BERT model that has seen Haitian Creole.
MODEL_ID = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = None

def get_bert_feature(text, word2ph, device=None):
    """
    Extracts BERT-based features from Haitian Creole text and aligns them to phonemes.
    
    - Text is tokenized with the multilingual BERT tokenizer.
    - The output hidden states (using, for example, the second-to-last layer)
      are repeated according to the provided `word2ph` mapping.
    """
    global model
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        # Ensure inputs are moved to the designated device.
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        output = model(**inputs, output_hidden_states=True)
        # Take the second-to-last hidden layer as features.
        hidden_states = output.hidden_states
        # The following mimics the French version’s approach.
        res_cat = torch.cat(hidden_states[-3:-2], dim=-1)[0].cpu()
    
    # Check the mapping length against BERT tokens.
    print(inputs["input_ids"].shape[-1] )
    print(word2ph)
    assert inputs["input_ids"].shape[-1] == len(word2ph), "Token count and word2ph mapping mismatch"
    phone_level_feature = []
    for i, count in enumerate(word2ph):
        # Repeat the embedding for the corresponding number of phonemes.
        repeated = res_cat[i].repeat(count, 1)
        phone_level_feature.append(repeated)
    
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T