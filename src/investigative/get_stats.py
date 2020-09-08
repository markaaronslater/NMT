#from processCorpuses import load_docs, to_sentences, filterSentences, normalizeCorpuses, createNamesTable
from collections import Counter, defaultdict
import re
import statistics
from unicodedata import normalize
from pickle import load, dump, HIGHEST_PROTOCOL










# compute histogram of sentence lengths of src and trg corpuses, in order to determine appropriate universal pad size,
# mask ranges, etc.
def get_lengths_histogram(src_sentences, trg_sentences):
    src_lengthCounts = Counter()
    for src_sentence in src_sentences:
        src_lengthCounts.update([len(src_sentence.split())]) #num tokens, not chs...

    # avg src_length

    trg_lengthCounts = Counter()
    for trg_sentence in trg_sentences:
        trg_lengthCounts.update([len(trg_sentence.split())])
    print("src sentence length counts:")
    for item in sorted(src_lengthCounts.items(), key = lambda pair: pair[0]):
        print(item)
    print()
    print("trg sentence length counts:")
    for item in sorted(trg_lengthCounts.items(), key = lambda pair: pair[0]):
        print(item)


def compute_length_stats(src_sentences, trg_sentences):
    src_lengths = []
    trg_lengths = []
    diffs = []
    pairs = zip(src_sentences, trg_sentences)
    for src_sent, trg_sent in pairs:
        len_s = len(src_sent)
        len_t = len(trg_sent)
        src_lengths.append(len_s)
        trg_lengths.append(len_t)
        diffs.append(len_s - len_t)
    print("avg src length: {}, avg trg length {}, avg diff {}".format(statistics.mean(src_lengths), statistics.mean(trg_lengths), statistics.mean(diffs)))
    print("median src length: {}, median trg length {}, median diff {}".format(statistics.median(src_lengths), statistics.median(trg_lengths), statistics.median(diffs)))




def get_songs(sentences):
    numSent = 0 # num sentences not ending in .
    
    for i, sent in enumerate(sentences):
        if '♫' in sent or '♪' in sent:
            print("{} {}".format(i, sent))
            print()
            numSent += 1
    print(numSent)