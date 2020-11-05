def build_subword_vocabs(corpus_path, vocab_type, vocab_threshold,
                            src_vocab_file, trg_vocab_file):
    if vocab_type == "subword_ind":
        vocabs = build_subword_ind_vocabs(corpus_path, vocab_type, vocab_threshold, src_vocab_file, trg_vocab_file)
    elif vocab_type == "subword_joint":
        vocabs = build_subword_joint_vocabs(corpus_path, vocab_type, vocab_threshold, src_vocab_file, trg_vocab_file)
    
    return vocabs

# in this context, "word" can refer to word, subword, or single character.
# (named this way for consistency with word-level vocab).
def build_subword_ind_vocabs(corpus_path, vocab_type, vocab_threshold,
                            src_vocab_file, trg_vocab_file):
    src_vocab = {'<pad>':0} 
    get_vocab_mapping(src_vocab, corpus_path + src_vocab_file, vocab_threshold)
    trg_vocab = {'<pad>':0, '<sos>':1, '<eos>':2}
    get_vocab_mapping(trg_vocab, corpus_path + trg_vocab_file, vocab_threshold)

    return {    "src_word_to_idx":src_vocab,
                "idx_to_src_word":{v:k for k, v in src_vocab.items()},
                "trg_word_to_idx":trg_vocab,
                "idx_to_trg_word":{v:k for k, v in trg_vocab.items()}
            }


def build_subword_joint_vocabs(corpus_path, vocab_type, vocab_threshold,
                                src_vocab_file, trg_vocab_file):
    vocab = {'<pad>':0, '<sos>':1, '<eos>':2}
    get_vocab_mapping(vocab, corpus_path + src_vocab_file, vocab_threshold)
    get_vocab_mapping(vocab, corpus_path + trg_vocab_file, vocab_threshold)

    #print(f"vocab: {vocab}")

    # pass 2 copies of each dict for compatibility with preprocess.py.
    # !!!change so not necessary.
    return {    "src_word_to_idx":vocab,
                "idx_to_src_word":{v:k for k, v in vocab.items()},
                "trg_word_to_idx":vocab,
                "idx_to_trg_word":{v:k for k, v in vocab.items()}
           }


def get_vocab_mapping(vocab, vocab_file, vocab_threshold):
    with open(vocab_file, "r") as f:
        for subword_count in f:
            subword, count = subword_count.split()[0], int(subword_count.split()[1])
            # if is a single character (either word-final or word-internal,
            # include in vocabulary even if does not occur above threshold).
            is_single_char = len(subword) == 1 or len(subword) == 3 and subword[-2:] == '@@'
            if subword not in vocab and (count >= vocab_threshold or is_single_char):
                vocab[subword] = len(vocab)