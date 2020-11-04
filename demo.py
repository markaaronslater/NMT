import torch
from subword_nmt.apply_bpe import BPE

from src.preprocessing.truecase import truecase_sentence, should_lower_de
from src.predict import predict

# wrapper that performs entire preprocessing, prediction, and postprocessing
# pipelines on a single user-inputted German sentence.
# all functions expect batches of sentences, so must adjust sentence
# so that conforms to this spec.
# download and initialize Stanza processors in Colab notebook, and pass as param.
# translator is pre-trained NMT model.

### change so that does list of sentences. to show off inference speed
# input is list of user-provided German sentences as strings 
def translate_single_sentence(input, stanza_processor, translator, bpe, idx_to_trg_word,
            device='cuda:0'):

    doc = stanza_processor('\n\n'.join(input)) # returns Document object
    sentences = doc.sentences[0] # extract Sentence object from singleton list of Sentences 
    truecased_sentences = []
    for sent in sentences:
        truecase_sentence(sent, should_lower_de)
        truecased_sentences.append(' '.join([word.text for word in sent.words]))

    
    #subword_vocab is a set, not a dict...
    #### apply learned bpe vocab to the sentence
    subword_sentences = [bpe.process_line(sent) for sent in truecased_sentences]


    # apply vocab

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






def get_subword_vocab(corpus_path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', 
            codes='bpe_codes', src_vocab_file='vocab.de'):

    vocabs = build_subword_vocabs(corpus_path, "subword_joint", hyperparams["vocab_threshold"], src_vocab_file, trg_vocab_file)

    # initalize BPE object
    bpe = BPE(codes, vocab=subword_vocab) 



    return set(vocab)