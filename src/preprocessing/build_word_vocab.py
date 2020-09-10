from collections import Counter










# wrapper function that given corpuses, learns a word-level vocab based on the src and trg train corpuses, that uses some trim function so that all words that do not fit in the vocabulary will be replaced in the corpus by the unknown token, <unk>. and then replaces each word with its index in its corresponding vocab
# corpuses is List[List[str]], where each sentence is List[str] (i.e., they are already tokenized).
# to build a higher quality vocab, they should also be intelligently decased so that, e.g., do not produce 2 entries for 'Hey' and 'hey', just bc occurred at beginning of some sentence, quotation, etc.
#??change so uses closures??
def get_word_vocabs(corpuses, trim_type="top_k"):
    assert trim_type == "top_k" or trim_type == "threshold"

    src_counter, trg_counter = get_train_word_counts(corpuses)
    # print_vocab(src_counter)
    # print_vocab(trg_counter)

    if trim_type == "top_k":
        src_vocab_words, trg_vocab_words = trim_vocab_by_topk(src_counter, trg_counter, src_k=30000, trg_k=30000)
    elif trim_type == "threshold":
        src_vocab_words, trg_vocab_words = trim_vocabs_by_thres(src_counter, trg_counter, src_thres=2, trg_thres=2)
    
    # print_vocab(src_vocab_words)
    # print_vocab(trg_vocab_words)

    vocabs = get_vocab_mappings(corpuses, src_vocab_words, trg_vocab_words)
    # print_vocab(vocabs["src_word_to_idx"])
    # print_vocab(vocabs["idx_to_src_word"])
    # print_vocab(vocabs["trg_word_to_idx"])
    # print_vocab(vocabs["idx_to_trg_word"])

    return vocabs


# determine the frequencies of words in the src and trg corpuses
def get_word_counts(corpus):
    counter = Counter()
    for sentence in corpus:
        counter.update(sentence)

    return counter


def get_train_word_counts(corpuses):
    src_counter = get_word_counts(corpuses["train.de"])
    print(f"length of src vocab before trimming: {len(src_counter.items())}")
    
    trg_counter = get_word_counts(corpuses["train.en"])
    print(f"length of trg vocab before trimming: {len(trg_counter.items())}")

    return src_counter, trg_counter


# keep only the src and trg words that occurred over the src and trg threshold frequencies, respectively
def trim_vocabs_by_thres(src_counter, trg_counter, src_thres=2, trg_thres=2):
    src_vocab_words = trim_vocab_by_thres(src_counter, src_thres)
    print(f"length of src vocab after trimming: {len(src_vocab_words)}")

    trg_vocab_words = trim_vocab_by_thres(trg_counter, trg_thres)
    print(f"length of trg vocab after trimming: {len(trg_vocab_words)}")

    return src_vocab_words, trg_vocab_words


def trim_vocab_by_thres(counter, thres=2):
    return set([word for word, count in counter.items() if count >= thres])


# keep only the top srcK most frequent words of src vocab and trgK most frequent words of trg vocab
def trim_vocabs_by_topk(src_counter, trg_counter, src_k=30000, trg_k=30000):
    src_vocab_words = trim_vocab_by_topk(src_counter, src_k)
    print(f"length of src vocab after trimming: {len(src_vocab_words)}")

    trg_vocab_words = trim_vocab_by_topk(trg_counter, trg_k)
    print(f"length of trg vocab after trimming: {len(trg_vocab_words)}")

    return src_vocab_words, trg_vocab_words


def trim_vocab_by_topk(counter, top_k=30000):
    return set([word for word, count in sorted(counter.items(), key = lambda c: c[1], reverse=True)[:top_k]])




# src and trg corpuses now consist exclusively of words in src and trg vocab_words, and <unk> 
# construct mappings from words to indices, and vice-versa.
def get_vocab_mappings(corpuses, src_vocab_words, trg_vocab_words):
    src_word_to_idx, idx_to_src_word = get_vocab_mapping(src_vocab_words, offset=2)
    trg_word_to_idx, idx_to_trg_word = get_vocab_mapping(trg_vocab_words, offset=4)

    return {    "src_word_to_idx":src_word_to_idx,
                "idx_to_src_word":idx_to_src_word, "trg_word_to_idx":trg_word_to_idx, "idx_to_trg_word":idx_to_trg_word
            }


# want special tokens to be at beginning of vocabs, but:
# src vocab uses 2 special tokens (so use offset 2 for remaining word indices)
# trg vocab uses 4 special tokens
def get_vocab_mapping(vocab_words, offset=2):
    word_to_idx = {word:i+offset for i, word in enumerate(vocab_words)}
    word_to_idx["<pad"], word_to_idx["<unk"] = 0, 1
    if offset == 4:
        # only trg sentences have start-of-sentence / end-of-sentence tokens
        word_to_idx["<sos"], word_to_idx["<eos"] = 2, 3
    idx_to_word = {v:k for k, v in word_to_idx.items()}

    return word_to_idx, idx_to_word


# print first <num> entries of vocab <v>
def print_vocab(v, num=10):
    for i, word in enumerate(v):
      if i == num:
        break
      print(word, end=' ')
    print()