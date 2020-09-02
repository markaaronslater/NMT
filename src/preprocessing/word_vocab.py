from collections import Counter

# determine the frequencies of words in the src and trg corpuses
def to_counters(corpuses):
    src_counter = Counter()
    for src_sentence in corpuses["train.de"]:
        src_counter.update(src_sentence)

    trg_counter = Counter()
    for trg_sentence in corpuses["train.en"]:
        trg_counter.update(trg_sentence)

    print("lengths of vocabs before trimming:", end=" ")
    print(f"src: {len(src_counter.items())}, trg: {len(trg_counter.items())}")
    
    return src_counter, trg_counter


# keep only the src and trg words that occurred over the src and trg threshold frequencies, respectively
def trim_vocabs(src_counter, trg_counter, srcThres=2, trgThres=2):
    srcwords = [word for word, count in src_counter.items() if count >= srcThres]
    trimmed_src_vocab = set(srcwords)

    trgwords = [word for word, count in trg_counter.items() if count >= trgThres]
    trimmed_trg_vocab = set(trgwords)

    print("lengths of vocabs after trimming: src: %d, trg: %d" % (len(trimmed_src_vocab), len(trimmed_trg_vocab)))
    return trimmed_src_vocab, trimmed_trg_vocab


# keep only the top srcK most frequent words of src vocab and trgK most frequent words of trg vocab
def trim_vocabs2(src_counter, trg_counter, srcK=30000, trgK=30000):
    sorted_src_counter = sorted(src_counter.items(), key = lambda c: c[1], reverse=True)
    topKsrcwordcountpairs = sorted_src_counter[:srcK]
    topKsrcwords = [word for word, count in topKsrcwordcountpairs]
    trimmed_src_vocab = set(topKsrcwords)

    sorted_trg_counter = sorted(trg_counter.items(), key = lambda c: c[1], reverse=True)
    topKtrgwordcountpairs = sorted_trg_counter[:trgK]
    topKtrgwords = [word for word, count in topKtrgwordcountpairs]
    trimmed_trg_vocab = set(topKtrgwords)

    print("lengths of vocabs after trimming:", end=" ")
    print(f"src: {len(trimmed_src_vocab)}, trg: {len(trimmed_trg_vocab)}")
    print()

    return trimmed_src_vocab, trimmed_trg_vocab


# construct mappings from words to indices, and vice-versa.
# consist of the special tokens, as well as all words that remain after trimming the vocabs
def computeVocabs(vocabs, corpuses):
    srcV = vocabs["srcV"]
    trgV = vocabs["trgV"]

    for sent in corpuses["train.de"]:
        for word in sent:
            if word not in srcV:
                srcV[word] = len(srcV)

    vocabs["idx_to_src_word"] = {srcV[key]:key for key in srcV}

    for sent in corpuses["train.en"]:
        for word in sent:
            if word not in trgV:
                trgV[word] = len(trgV)
    
    vocabs["idx_to_trg_word"] = {trgV[key]:key for key in trgV}


# print first <num> entries of vocab <v>
def print_vocab(v, num=10):
    for i, word in enumerate(v):
      if i == num:
        break
      print(word, end=' ')
    print()


# 'unknown' tokens are only necessary when using a word-level (rather than subword-level) vocabulary
def replace_with_unk_tokens(corpuses, trimmed_src_vocab, trimmed_trg_vocab):     
    removeOOV(corpuses["train.de"], trimmed_src_vocab)    
    removeOOV(corpuses["train.en"], trimmed_trg_vocab)
    # next 2 are no-ops if these corpuses dne (e.g., when debugging):    
    removeOOV(corpuses["dev.de"], trimmed_src_vocab)  
    # (do not replace dev targets with unk) 
    removeOOV(corpuses["test.de"], trimmed_src_vocab)


# for each word, if it does not belong to the trimmed vocabulary (it is an Out-Of-Vocabulary word), replace it with the 'unknown' token
def removeOOV(sentences, trimmed_vocab):
    for i, sent in enumerate(sentences):
        sentences[i] = [token if token in trimmed_vocab else '<unk>' for token in sent]