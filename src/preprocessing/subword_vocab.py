# vocab files contain lines of the form "<subword> <count>"
# i assume i'm supposed to conflate the same morpheme of both langs into single entry/meaning
def computeBPEvocabs(src_vocab_file, trg_vocab_file):
    vocab = {'<pad>':0, '<sos>':1, '<eos>':2}
    with open(src_vocab_file, "r") as f:
        #!!! ??do i need to use utf8??
        for line in f:
            subword = line.split()[0]
            if subword not in vocab:
                vocab[subword] = len(vocab)
    numDe = len(vocab)-3
    print("number of german subwords is {}".format(numDe))
    with open(trg_vocab_file, "r") as f:
        for line in f:
            subword = line.split()[0]
            if subword not in vocab:
                vocab[subword] = len(vocab)

    numEn = len(vocab) - numDe - 3

    # this highly underreports bc of subwords that belong to both langs
    # not getting added twice:
    print("number of english subwords is {}".format(numEn))

    print("total number of subwords is {}".format(len(vocab)))

    idx_to_subword = dict((v,k) for (k,v) in vocab.items())
    return vocab, idx_to_subword
