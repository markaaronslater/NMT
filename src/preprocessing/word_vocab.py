from collections import Counter

from preprocess import replace_with_indices, to_indices

def get_normalized_corpuses():
    get_processed_corpuses()
    decase_corpuses()
    if vocab_type == "subword":
        get_subword_corpuses()

    # !!!save normalized corpuses to pickle files

    return normalized_corpuses

# does not make a ton of sense to make one end-to-end preprocessing function, bc the stanford nlp stanza pos-tagger, tokenizer, segmenter step is time-consuming, so makes more sense to do that a single time & save the results to files.
# next, use the processor outputs to normalize the corpuses.
# finally, convert the normalized corpuses into batches of tensors that can directly pass to the model.

# convert corpuses into model inputs:
# (1) run segment_tokenize_tag # will only ever need to run this once
# (2) run get_normalized_corpuses()
# (3) run produce_network_inputs()

# -> after each step, saves to pickle files so never need repeat a step


# train model:
# (4) build model using a config file or by passing arguments via argparse

# test model:
# (5) choose a model checkpoint, perform inference on dev set
# (6) eval using sacrebleu script




# TODO: wrap these up in single function, get_subword_corpuses():
# (optional) if using subword embeddings:
#   -run jointBPE.sh on these decased files. 
#   -read these bpe_ files into corpuses.
#   -tokenize via naive whitespace splitting (do not need to run stanfordnlpprocessor(segment, tokenize) again, bc essentially already tokenized as sentences of subwords, since jointBPE.sh was applied to tokenized sentences of words). 
#   -save to pickle files (such that filename documents choices of numMerges, vocabThreshold, etc.)

# should also offer a shortcut so that if already have bpe corpuses ready, can just directly load them from a pickle file



# (4) run produce_network_inputs()


# expects each listed file to consist of one sentence per line, where line i of trg corpus is the translation of line i of src corpus
# stores processor outputs to pickle files



# given that corpuses consists of tokenized, segmented, decased sentences, 
# (if word_level vocabs, then lists of words)
# (if subword_level vocabs, then lists of subwords)
#   must have already:
#       applied stanfordnlpprocessor to first tokenize, segment, and pos-tag at word-level, so that can properly decase. then, write to decased_ variants of corpus files. then, run jointBPE.sh on these decased files. then, read these files into corpuses. tokenize via naive whitespace splitting (do not need to run stanfordnlpprocessor(segment, tokenize) again, bc essentially already tokenized as sentences of subwords, since jointBPE.sh was applied to tokenized sentences of words). lastly, save to pickle files such that filename documents choices of numMerges, vocabThreshold, etc.


# produces list of batches of tensors that can be directly fed to the encoder-decoder network
# !!!use argparse
def produce_network_inputs():
    assert vocab_type == "word" or vocab_type == "subword"

    # load normalized corpuses from pickle files
    corpuses = load_normalized_corpuses()

    # build vocabs
    if vocab_type == "word":
        get_word_vocabs()
        replace_with_unk_tokens(corpuses, src_vocab_words, trg_vocab_words)
    elif vocab_type == "subword":
        # get_vocabs is much simpler than word_level, bc all the work was already done by jointBPE.sh. just needs to read it from file.
        get_subword_vocabs()


    get_references(corpuses) # for estimating model quality after each epoch using corpus_bleu

    add_start_end_tokens(corpuses) # dont include in references

    replace_with_indices(corpuses, vocabs)

    get_batches()


    # !!!save batches and references to pickle files so can skip this step next time

    # wrt vocabulary, model just needs idx_to_trg_word for producing translations, and needs lengths of vocabs so can build embedding table
    return batches, vocabs["idx_to_trg_word"]



# wrapper function that given corpuses, learns a word-level vocab based on the src and trg train corpuses, that uses some trim function so that all words that do not fit in the vocabulary will be replaced in the corpus by the unknown token, <unk>. and then replaces each word with its index in its corresponding vocab

# corpuses is List[List[str]], where each sentence is List[str] (i.e., they are already tokenized), and if it is a target sentence, begins with <sos> and ends with <eos>. 
# to build a higher quality vocab, sentences of corpuses should also be decased so that, e.g., do not produce 2 entries for 'Hey' and 'hey', just bc occurred at beginning of some sentence, quotation, etc.

#??change so uses closures??
def get_word_vocabs(corpuses, trim_type="top_k"):
    assert trim_type == "top_k" or trim_type == "threshold"

    src_counter, trg_counter = get_train_word_counts(corpuses)
    print_vocab(src_counter)
    print_vocab(trg_counter)

    if trim_type == "top_k":
        src_vocab_words, trg_vocab_words = trim_vocab_by_topk(src_counter, trg_counter, src_k=30000, trg_k=30000)
    elif trim_type == "threshold":
        src_vocab_words, trg_vocab_words = trim_vocabs_by_thres(src_counter, trg_counter, src_thres=2, trg_thres=2)
    
    print_vocab(src_vocab_words)
    print_vocab(trg_vocab_words)


    vocabs = get_vocab_mappings(corpuses, src_vocab_words, trg_vocab_words)
    
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




# 'unknown' tokens are only necessary when using a word-level (rather than subword-level) vocabulary
def replace_with_unk_tokens(corpuses, src_vocab_words, trg_vocab_words):     
    removeOOV(corpuses["train.de"], src_vocab_words)    
    removeOOV(corpuses["train.en"], trg_vocab_words)
    # next 2 are no-ops if these corpuses dne (e.g., when debugging):    
    removeOOV(corpuses["dev.de"], src_vocab_words)  
    # (do not replace dev targets with unk) 
    removeOOV(corpuses["test.de"], src_vocab_words)


# for each word, if it does not belong to the trimmed vocabulary (it is an Out-Of-Vocabulary word), replace it with the 'unknown' token
def removeOOV(sentences, vocab_words):
    for i, sent in enumerate(sentences):
        sentences[i] = [token if token in vocab_words else '<unk>' for token in sent]


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