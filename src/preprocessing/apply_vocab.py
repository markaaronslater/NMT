
# inputs: have already built the vocabs, so replace each word of corpus with the idx it maps to in the vocab.
# corpuses: List[List[str]]
# (already has sos and eos tags)

# vocabs: Dict[str, Dict[str, int]]

# output: None (modifies each corpus in-place)
def apply_vocab(corpuses, vocabs, vocab_type="word"):
    if vocab_type == "word":
        replace_with_unk_tokens(corpuses, vocabs["src_word_to_idx"], vocabs["trg_word_to_idx"])
    replace_with_indices(corpuses, vocabs)


# 'unknown' tokens are only necessary when using a word-level (rather than subword-level) vocabulary
# vocab can be either a set of words or a word-to-idx mapping (i am using the latter)
def replace_with_unk_tokens(corpuses, src_vocab, trg_vocab):     
    removeOOV(corpuses["train.de"], src_vocab)    
    removeOOV(corpuses["train.en"], trg_vocab)
    # next 2 are no-ops if these corpuses dne (e.g., when debugging):    
    removeOOV(corpuses["dev.de"], src_vocab)  
    # (do not replace dev targets with unk) 
    removeOOV(corpuses["test.de"], src_vocab)


# for each word, if it does not belong to the trimmed vocabulary (it is an Out-Of-Vocabulary word), replace it with the 'unknown' token
def removeOOV(sentences, vocab):
    for i, sent in enumerate(sentences):
        sentences[i] = [token if token in vocab else '<unk>' for token in sent]


def replace_with_indices(corpuses, vocabs):
    to_indices(corpuses["train.de"], vocabs["src_word_to_idx"])
    to_indices(corpuses["train.en"], vocabs["trg_word_to_idx"])
    to_indices(corpuses["dev.de"], vocabs["src_word_to_idx"])   
    to_indices(corpuses["test.de"], vocabs["src_word_to_idx"])


def to_indices(sentences, vocab):
    for i, sent in enumerate(sentences):
        new_sent = [] # contains corresponding indices of sent
        for word in sent:
            try:
                new_sent.append(vocab[word])
            except KeyError:
                ### not quite sure why this happens. maybe some sort of quirk in the subword-nmt script. i'll figure it out later.
                # one clue is that they all seem to be special unicode ch's. maybe i am not using proper encoding scheme.
                print(f"warning: found and removed unknown word: {word} in sent: {sent}")
                print()
        sentences[i] = new_sent