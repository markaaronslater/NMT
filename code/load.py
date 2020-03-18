from pickle import load

#### before ever running this program, you must first run processAndStoreCorpuses.py to put the preprocessed corpuses
#### in the appropriate files

def load_corpuses(preprocessed_srcFile, preprocessed_trgFile):
    src_sentences = load(open(preprocessed_srcFile, 'rb'))
    trg_sentences = load(open(preprocessed_trgFile, 'rb'))

    return src_sentences, trg_sentences


def load_mappings(srcVFile, trgVFile, idx_to_src_wordFile, idx_to_trg_wordFile):
    with open(srcVFile, 'rb') as handle:
        srcV = load(handle)
    with open(trgVFile, 'rb') as handle:
        trgV = load(handle)
    with open(idx_to_src_wordFile, 'rb') as handle:
        idx_to_src_word = load(handle)
    with open(idx_to_trg_wordFile, 'rb') as handle:
        idx_to_trg_word = load(handle)

    return srcV, trgV, idx_to_src_word, idx_to_trg_word


# print first and last 5 sentences of loaded corpus in word (rather than index) form
def checkSentences(src_sentences, trg_sentences, idx_to_src_word, idx_to_trg_word):
    print("num src sentences", len(src_sentences))
    print("num trg sentences", len(trg_sentences))

    print("first 5 src sentences:")
    for i in range(5):
        print(' '.join([idx_to_src_word[idx] for idx in src_sentences[i]]))
    print()
    print("first 5 trg sentences:")
    for i in range(5):
        print(' '.join([idx_to_trg_word[idx] for idx in trg_sentences[i]]))
    print()
    print("last 5 src sentences:")
    for i in range(len(src_sentences) - 5, len(src_sentences)):
        print(' '.join([idx_to_src_word[idx] for idx in src_sentences[i]]))
    print()
    print("last 5 trg sentences:")
    for i in range(len(trg_sentences) - 5, len(trg_sentences)):
        print(' '.join([idx_to_trg_word[idx] for idx in trg_sentences[i]]))
    print()


def checkMappings(srcV, trgV, idx_to_src_word, idx_to_trg_word):
    print("length of srcV:", len(srcV))
    print("length of trgV:", len(trgV))
    print("length of idx_to_src_word:", len(idx_to_src_word))
    print("length of idx_to_trg_word:", len(idx_to_trg_word))