import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pickle import load, dump, HIGHEST_PROTOCOL
from collections import Counter
from random import shuffle
from random import randint
from nltk.translate.bleu_score import sentence_bleu
import re
import string
##### Performs all preprocessing and stores results in series of files that are directly loaded by encoderdecoder



# load the corpus files into strings and then converts each into list of sentences
# corpuses is a dictionary where the keys are file names, and the values are None. afterward, the values are the string form of their text
def load_docs(path, corpuses, bpe=False, tok=False, decased=False):
    prefix = ''
    infix = ''

    if bpe:
        infix = '.BPE'
    elif tok:
        prefix = 'tok_'
    elif decased:
        prefix = 'decased_'

    ###!!!figure out elegant way to handle infix, or convert BPE to a prefix
    for corpus_name in corpuses:
        with open(path + prefix + corpus_name, mode='rt', encoding='utf-8') as f:
            corpuses[corpus_name] = f.read().strip()
            if corpuses[corpus_name]: 
                corpuses[corpus_name] = corpuses[corpus_name].split('\n')
            else:
                corpuses[corpus_name] = [] # f was empty


# print out first <num> sentences of each corpus
def print_corpuses(corpuses, num=10):
    for corpus_name in corpuses:
        corpus = corpuses[corpus_name]
        for sent in corpus[:num]:
            print(sent)
        print()
    



# corpuses is a list of lists of str(sentences)
# applies naive true-casing (lower case 1st word of sentence and any 
# word following a double-quote ??what about :??), and tokenization



def normalizeCorpuses(path, corpuses):
    #load_docs(path, corpuses)
    # print("texts:")
    # print(texts)
    # print()
    # stage 0-remove songs and sentences over 100 words in length from train sets
    #texts[0], texts[1] = filterSentences(texts[0], texts[1])
    
    ###!!!fix this with iwslt-en-de2
    #filterSentences(corpuses)
    filter_corpuses(corpuses)


    # stage 1-tokenize corpuses -> produces references. for De -> En, only 
    # necessary for train_trg (when debugging new nets, i.e., overfitting 
    # to trainset) and dev_trg (when estimating model's translation ability
    # after finish an epoch)
    #tok_texts = tokenizeCorpuses(path, corpuses)
    tokenize_corpuses(path, corpuses)



    # print("tok_texts:")
    # print(tok_texts)
    # print()
    # stage 2-create names tables -> produces dictionaries of words that 
    # should not be lowercased
    #de_namesTable, _ = createNamesTable(path, "names.de", tok_texts[0][:])
    #en_namesTable, _ = createNamesTable(path, "names.en", tok_texts[1][:])

    # stage 3-decase corpuses -> produces training corpuses. for De -> En, only 
    # necessary for train_src, train_trg, dev_src and test_src
    ### ???wait, why did i do this for the dev and test source sentences, too. isnt that cheating???
    #decased_texts = decaseCorpuses(tok_texts, path, de_namesTable, en_namesTable)
    # print("decased_texts:")
    # print(decased_texts)
    # print()
    #return decased_texts



























# for typical corpuses, e.g., train and dev sets, converts a list of sentences as strings into a list of sentences as lists of words
# list of str(sentence)'s to list of lists of words.

# for reference corpuses (used by corpus_bleu evaluation), converts a list of sentences as strings into a list of reference sets, where a reference set is a  list of acceptable translations for the corresponding source , where a translation is a list of words of the target sentence.
# (in this project, there will always be a single reference translation for each source sentence, so each reference set is a singleton list)
def str_to_list(corpuses, ref=False):
    for corpus_name in corpuses:
        corpus = corpuses[corpus_name]
        for i, sent in enumerate(corpus):
            corpus[i] = sent.split() if not ref else [sent.split()]


# prepend and append trg sentences with start-of-sentence token, <sos>, and end-of-sentence token, <eos>, respectively
def add_start_end_tokens(corpuses):
    ###???is this list concat a constant time op??? is it creating a copy???
    corpuses["train.en"] = [['<sos>'] + sent + ['<eos>'] for sent in corpuses["train.en"]]


def replace_with_indices(corpuses, vocabs):
    to_indices(corpuses["train.de"], vocabs["srcV"])
    to_indices(corpuses["train.en"], vocabs["trgV"])
    to_indices(corpuses["dev.de"], vocabs["srcV"])   
    to_indices(corpuses["test.de"], vocabs["srcV"])


def to_indices(sentences, vocab):
    idx_sentences = []
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



