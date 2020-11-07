from collections import Counter
# given corpuses, learns a word-level vocab based
# on the src and trg train corpuses, that uses some trim function so
# that all words that do not fit in the vocabulary will be replaced in the
# corpus by the unknown token, <unk>.
def build_word_vocabs(corpuses, hyperparams):
    src_counter, trg_counter = get_train_word_counts(corpuses)
    if hyperparams["trim_type"] == "top_k":
        src_vocab_words, trg_vocab_words = trim_vocabs_by_topk(src_counter, trg_counter, hyperparams["src_k"], hyperparams["trg_k"])
    elif hyperparams["trim_type"] == "threshold":
        src_vocab_words, trg_vocab_words = trim_vocabs_by_thres(src_counter, trg_counter, hyperparams["src_thres"], hyperparams["trg_thres"])

    vocabs = get_vocab_mappings(corpuses, src_vocab_words, trg_vocab_words)

    return vocabs


def get_train_word_counts(corpuses):
    src_counter = get_word_counts(corpuses["train.de"])
    print(f"length of src vocab before trimming: {len(src_counter.items())}")
    
    trg_counter = get_word_counts(corpuses["train.en"])
    print(f"length of trg vocab before trimming: {len(trg_counter.items())}")

    return src_counter, trg_counter


def get_word_counts(corpus):
    counter = Counter()
    for sentence in corpus:
        counter.update(sentence)

    return counter


# keep only the src and trg words that occurred over the src and trg
# threshold frequencies, respectively.
def trim_vocabs_by_thres(src_counter, trg_counter, src_thres=2, trg_thres=2):
    src_vocab_words = trim_vocab_by_thres(src_counter, src_thres)
    trg_vocab_words = trim_vocab_by_thres(trg_counter, trg_thres)

    return src_vocab_words, trg_vocab_words


def trim_vocab_by_thres(counter, thres=2):
    return set([word for word, count in counter.items() if count >= thres])


# keep only the top src_k most frequent words of src vocab and trg_k
# most frequent words of trg vocab.
def trim_vocabs_by_topk(src_counter, trg_counter, src_k=30000, trg_k=30000):
    src_vocab_words = trim_vocab_by_topk(src_counter, src_k)
    trg_vocab_words = trim_vocab_by_topk(trg_counter, trg_k)

    return src_vocab_words, trg_vocab_words


def trim_vocab_by_topk(counter, top_k=30000):
    return set([word for word, count in sorted(counter.items(), key = lambda c: c[1], reverse=True)[:top_k]])


# construct mappings from words to indices, and vice-versa, so can
# later replace words of corpuses with their indices.
def get_vocab_mappings(corpuses, src_vocab_words, trg_vocab_words):
    src_word_to_idx, idx_to_src_word = get_vocab_mapping(src_vocab_words, offset=2)
    trg_word_to_idx, idx_to_trg_word = get_vocab_mapping(trg_vocab_words, offset=4)

    return {    "src_word_to_idx":src_word_to_idx,
                "idx_to_src_word":idx_to_src_word,
                "trg_word_to_idx":trg_word_to_idx,
                "idx_to_trg_word":idx_to_trg_word
            }


def get_vocab_mapping(vocab_words, offset=2):
    word_to_idx = {word:i+offset for i, word in enumerate(vocab_words)}
    word_to_idx["<pad>"], word_to_idx["<unk>"] = 0, 1
    if offset == 4:
        # only trg sentences have start-of-sentence / end-of-sentence tokens
        word_to_idx["<sos>"], word_to_idx["<eos>"] = 2, 3
    idx_to_word = {v:k for k, v in word_to_idx.items()}

    return word_to_idx, idx_to_word