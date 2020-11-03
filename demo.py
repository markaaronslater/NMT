import torch

from src.preprocessing.truecase import truecase_sentence, should_lower_de
from src.predict import predict

# wrapper that performs entire preprocessing, prediction, and postprocessing
# pipelines on a single user-inputted German sentence.
# all functions expect batches of sentences, so must adjust sentence
# so that conforms to this spec.
# download and initialize Stanza processors in Colab notebook, and pass as param.
# translator is pre-trained NMT model.
def translate_single_sentence(input, stanza_processor, translator, idx_to_trg_word, device):
    doc = stanza_processor(input) # returns Document object
    sent = doc.sentences[0] # extract Sentence object from singleton list of Sentences 
    truecase_sentence(sent, should_lower_de)
    sent = [word.text for word in sent.words]

    #### apply learned bpe vocab to the sentence

    src_len = len(sent)
    zero = torch.zeros(1, device=device, dtype=torch.long)
    encoder_inputs = {
        "in":torch.tensor(sent, device=device, dtype=torch.long).view(1, src_len),
        "sorted_lengths":torch.tensor(src_len, device=device, dtype=torch.long),
        "idxs_in_sorted":zero
    }
    decoder_inputs = {
        # use dummy mask of all zeros bc, trivially, no tokens of src sentence are padding. 
        "mask":torch.zeros(1, 1, src_len, device=device, dtype=torch.bool),
        "max_src_len": src_len
    }
    corpus_indices = zero

    test_batches = [(encoder_inputs, decoder_inputs, corpus_indices)]
    translations, _ = predict(translator, test_batches, idx_to_trg_word, write=False)
    translation = translations[0] # singleton batch of translations

    return translation