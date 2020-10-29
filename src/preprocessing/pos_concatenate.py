from apply_stanza_processors import retrieve_stanza_outputs

def pos_concatenate_corpuses(*corpus_names, path='/content/gdrive/My Drive/iwslt16_en_de/', vocab_type="word_pos"):
    assert vocab_type in ["word_pos", "subword_pos"]
    morpho_corpuses = retrieve_stanza_outputs(*corpus_names, path) # corpuses containing morphological data for each sentence

    # apply delimiter of 