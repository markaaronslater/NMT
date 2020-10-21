# vocab files contain lines of the form "<subword> <count>"
# i assume i'm supposed to conflate the same morpheme of both langs into single entry/meaning

# get_vocabs is much simpler than word_level, bc all the work was already done by jointBPE.sh. just needs to read it from file. i.e., don't need to pass corpuses as param
def build_subword_vocabs(src_vocab_file, trg_vocab_file):
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



if __name__ == '__main__':
    v, i = build_subword_vocabs('./Downloads/vocab.de', './Downloads/vocab-2.en')
    for key, val in v.items():
        print(f"{key}: {val}")