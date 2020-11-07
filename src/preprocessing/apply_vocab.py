# have already built the vocabs, so replace each word of corpus
# with the idx it maps to in the vocab.
# each entry of corpuses is a list of sentences, where sentence is list of words.
def apply_vocab(corpuses, vocabs, vocab_type="word"):
    if vocab_type == "word":
        replace_with_unk_tokens(corpuses, vocabs["src_word_to_idx"], vocabs["trg_word_to_idx"])
    replace_with_indices(corpuses, vocabs)


# 'unknown' tokens are only necessary when using a word-level
# (rather than subword-level) vocabulary.
def replace_with_unk_tokens(corpuses, src_vocab, trg_vocab):     
    removeOOV(corpuses["train.de"], src_vocab)    
    removeOOV(corpuses["train.en"], trg_vocab)
    if "dev.de" in corpuses:
        removeOOV(corpuses["dev.de"], src_vocab) 
    if "test.de" in corpuses:
        removeOOV(corpuses["test.de"], src_vocab)
    # (do not replace dev or test targets with unk) 


# for each word, if it does not belong to the trimmed vocabulary
# (it is an Out-Of-Vocabulary word), replace it with the 'unknown' token.
def removeOOV(sentences, vocab):
    for i, sent in enumerate(sentences):
        sentences[i] = [token if token in vocab else '<unk>' for token in sent]


def replace_with_indices(corpuses, vocabs):
    to_indices("train.de", corpuses["train.de"], vocabs["src_word_to_idx"])
    to_indices("train.en", corpuses["train.en"], vocabs["trg_word_to_idx"])
    if "dev.de" in corpuses:
        to_indices("dev.de", corpuses["dev.de"], vocabs["src_word_to_idx"])   
    if "test.de" in corpuses:
        to_indices("test.de", corpuses["test.de"], vocabs["src_word_to_idx"])


# except clause catches OOV symbols in src dev/test sets when using subword vocabs.
# there are 2 types of OOV symbols:
# 1 - a character that never occurred at all (whether in isolation or as part of
#     word) in src training set.
# 2 - a symbol that never occurred in isolation in the src training set after
#     segment it based on learned bpe codes (always occurred inside larger
#     symbol) (this includes the case of single characters).
#       -> it could have occurred in original train set, but not the threshold
#          number of times, as specified by <vocab_threshold>, so never occurred
#          in segmented train set.
def to_indices(corpus_name, sentences, vocab):
    for i, sent in enumerate(sentences):
        new_sent = [] # contains corresponding indices of sent
        for symbol in sent:
            # symbol is either a word or subword
            try:
                new_sent.append(vocab[symbol])
            except KeyError:
                #continue
                print(f"warning: found and removed unknown symbol {symbol} in following sentence of {corpus_name}: {sent}")
        sentences[i] = new_sent