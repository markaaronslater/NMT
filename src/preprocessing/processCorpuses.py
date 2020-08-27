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

# # pass path to toy corpuses
# def load_docs(path, bpe=False, tok=False, decased=False):
#     prefix = ''
#     infix = ''

#     if bpe:
#         infix = '.BPE'
#     elif tok:
#         prefix = 'tok_'
#     elif decased:
#         prefix = 'decased_'

#     train_srcFile = prefix + 'train' + infix + '.de'
#     train_trgFile = prefix + 'train' + infix + '.en'
#     # dev_srcFile = prefix + 'dev' + infix + '.de'
#     # dev_trgFile = prefix + 'dev' + infix + '.en'
#     # test_srcFile = prefix + 'test' + infix + '.de'

            
#     with open(path + train_srcFile, mode='rt', encoding='utf-8') as f:
#         train_srctext = f.read()

#     with open(path + train_trgFile, mode='rt', encoding='utf-8') as f:
#         train_trgtext = f.read()

#     # with open(path + dev_srcFile, mode='rt', encoding='utf-8') as f:
#     #     dev_srctext = f.read()

#     # with open(path + dev_trgFile, mode='rt', encoding='utf-8') as f:
#     #     dev_trgtext = f.read()
    
#     # with open(path + test_srcFile, mode='rt', encoding='utf-8') as f:
#     #     test_srctext = f.read()



#     texts = to_sentences([train_srctext, train_trgtext])
#     return texts





# load the corpus files into strings and then converts each into list of sentences
def load_docs(path, bpe=False, tok=False, decased=False):
    prefix = ''
    infix = ''

    if bpe:
        infix = '.BPE'
    elif tok:
        prefix = 'tok_'
    elif decased:
        prefix = 'decased_'

    train_srcFile = prefix + 'train' + infix + '.de'
    train_trgFile = prefix + 'train' + infix + '.en'
    dev_srcFile = prefix + 'dev' + infix + '.de'
    dev_trgFile = prefix + 'dev' + infix + '.en'
    test_srcFile = prefix + 'test' + infix + '.de'

            
    with open(path + train_srcFile, mode='rt', encoding='utf-8') as f:
        train_srctext = f.read()

    with open(path + train_trgFile, mode='rt', encoding='utf-8') as f:
        train_trgtext = f.read()

    with open(path + dev_srcFile, mode='rt', encoding='utf-8') as f:
        dev_srctext = f.read()

    with open(path + dev_trgFile, mode='rt', encoding='utf-8') as f:
        dev_trgtext = f.read()
    
    with open(path + test_srcFile, mode='rt', encoding='utf-8') as f:
        test_srctext = f.read()

    texts = to_sentences([train_srctext, train_trgtext, dev_srctext, dev_trgtext, test_srctext])
    return texts


# pass a list of texts, and produces list of lists of str(sentences)
def to_sentences(texts):
    return [text.strip().split('\n') for text in texts]
    #return [text.split('\n') for text in texts] # for debug, allow empty sent







# corpuses is a list of lists of str(sentences)
# applies naive true-casing (lower case 1st word of sentence and any 
# word following a double-quote ??what about :??), and tokenization


########### regular expressions, lookup tables, etc. ##############
###################################################################
###################################################################

# do not convert these to lowercase even if begin a sentence or pair of quotes:
# Iwords = ["I", "I'll", "I'm", "I'd", "I've"]
# -> now handled by namesTable

# decase the words that follow end of sentence symbols:

### !!!change this to eos_symbols, or something
eos = {}
for key in [".", "!", "?", ":", "..", "...", "...."]:
    eos[key] = 1
# acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
# -> now handled by namesTable




smartsinglequotes = re.compile(r'‘|’')
smartdoublequotes = re.compile(r'“|”')

# preceding the beginning single quote, there must be a non-word character (use assertion bc don't want it included in the match)
# following the beginning single quote, there must not be em or a digit
leftSinglequotes = re.compile(r'(?<=\W)\'(?!em|\d)(\w+)') 

# preceding the ending single quote, there must not be <in> or <s>
# following the ending single quote, there must be a non-word character (use assertion bc don't want it included in the match)
rightSinglequotes = re.compile(r'(\w+)(?<!in|.s)\'(?=\W)') 

# the naive tokenization/eos disambiguation step segments terminal periods
# from words, unless they also contain an internal period
# ex) it keeps U.S. intact, but segments Mr. into Mr .
# we stitch such acronyms back together after the fact.
# I wish I thought of a more elegant way, but it is naive, after all.
acronyms = re.compile(r'(Mr|Mrs|Dr|Ms|etc|ca|St|Mt|Lt)\s\.')
###################################################################
###################################################################
###################################################################



def normalizeCorpuses(path):
    texts = load_docs(path)
    # print("texts:")
    # print(texts)
    # print()
    # stage 0-remove songs and sentences over 100 words in length from train sets
    texts[0], texts[1] = filterSentences(texts[0], texts[1])

    # stage 1-tokenize corpuses -> produces references. for De -> En, only 
    # necessary for train_trg (when debugging new nets, i.e., overfitting 
    # to trainset) and dev_trg (when estimating model's translation ability
    # after finish an epoch)
    tok_texts = tokenizeCorpuses(texts, path)
    # print("tok_texts:")
    # print(tok_texts)
    # print()
    # stage 2-create names tables -> produces dictionaries of words that 
    # should not be lowercased
    de_namesTable, _ = createNamesTable(path, "names.de", tok_texts[0][:])
    en_namesTable, _ = createNamesTable(path, "names.en", tok_texts[1][:])

    # stage 3-decase corpuses -> produces training corpuses. for De -> En, only 
    # necessary for train_src, train_trg, dev_src and test_src
    ### ???wait, why did i do this for the dev and test source sentences, too. isnt that cheating???
    decased_texts = decaseCorpuses(tok_texts, path, de_namesTable, en_namesTable)
    # print("decased_texts:")
    # print(decased_texts)
    # print()
    return decased_texts

def tokenizeCorpuses(corpuses, path):
    corpus_names = ["train.de", "train.en", "dev.de", "dev.en", "test.de"]
    #corpus_names = ["train.de", "train.en"]

    tok_corpuses = []
    for corpus_idx, corpus in enumerate(corpuses):
        tok_corpus = []
        corpus_name = corpus_names[corpus_idx]
        with open(path + "tok_" + corpus_name, "w") as f:
            print("tokenizing " + corpus_name + "...")
            for sent in corpus:
                sent = naiveTokenize(sent)
                tok_corpus.append(sent)
                f.write(sent + '\n') #!!! this is why references diff length than trgs, bc adding a newline
            tok_corpuses.append(tok_corpus)

    return tok_corpuses


def decaseCorpuses(corpuses, path, de_namesTable, en_namesTable):
    corpus_names = ["train.de", "train.en", "dev.de", "dev.en", "test.de"]
    #corpus_names = ["train.de", "train.en"]

    decased_corpuses = []

    for corpus_idx, corpus in enumerate(corpuses):
        decased_corpus = []
        corpus_name = corpus_names[corpus_idx]
        namesDict = de_namesTable if corpus_name[-2:] == "de" else en_namesTable
    
        with open(path + "decased_" + corpus_name, "w") as f:
            print("decasing " + corpus_name + "...")
            for sent in corpus:
                sent = naiveDecase(sent, namesDict)
                decased_corpus.append(sent)
                f.write(sent + '\n')
            decased_corpuses.append(decased_corpus)
    
    return decased_corpuses



# receives str(sentence), returns tokenized str(sentence)
def naiveTokenize(sent):
    ### simplify smartquotes
    sent = re.sub(smartsinglequotes, '\'', sent)
    sent = re.sub(smartdoublequotes, '\"', sent)

    ### disambiguate single quotes from apostrophes
    sent = re.sub(leftSinglequotes, r"' \1", sent) # insert a space
    sent = re.sub(rightSinglequotes, r"\1 '", sent) # insert a space

    ### tokenize and segment sentences, keeping abbreviations and conjunctions, etc., together
    # abbr stands for abbreviation that contains internal periods. 
    #                       numbers                         abbr    ellipse      words            --         punctuation
    tokens = re.findall(r"(\.\d+|(?:\d+(?:[.,]\d+)*)(?:'?s|(?:th))?|\.\.\.*|\w+\.\w+[.\w]*|[\w']+|--|[.,:!?;\"\[\]\(\)$—-])", sent)
    sent = ' '.join(tokens)
    sent = re.sub(acronyms, r"\1.", sent) # Mr . -> Mr.

    return sent



def naiveDecase(sent, namesDict):
    if not sent:
        return sent # for debugging empty sentence

    sent = sent.split()
    #if sent[0] not in Iwords and sent[0] not in acList:

    # lower case first word of the sentence:
    if sent[0] not in namesDict:
        sent[0] = sent[0][0].lower() + sent[0][1:] # this handles leading acronyms, like people's names, followed by :
    # ex) BG -> bG, not bg, so that when predict this during inference, can recase to BG, not Bg
    ### (this format is common in the corpus, bc transcripts of TED-talks, where diff speakers will precede given sentences, in a dialogue exchange, etc.)

    positions = [i for i,word in enumerate(sent) if (word in eos or word == '"') and i != len(sent)-1]
    #quotePositions = [i for i,word in enumerate(sent) if word == '"' and i != len(sent)-1]
    # (do not extract double quote at last position of sentence, bc nothing follows it)
    
    # in decasing, no need to discriminate between quote openers and closers
    # lowercase the first word inside a pair of doublequotes,
    # and first word to follow a pair of doublequotes or a lone period, exclamation point, or question mark
    for j in positions: 
        if sent[j+1] not in namesDict:
            sent[j+1] = sent[j+1].lower()
    # makes simplifying assumption that for any sent with odd number of quotes,
    # the unpaired quote comes at the end, not the beginning
    # will be correct ~50% of the time, presumably.



    return ' '.join(sent)


# when passed list of str(words), capitalizes first word of sentence
# and the first word following a double quote
def naiveRecase(sent):
    if not sent:
        return sent # for debugging empty sentence

    try:
        sent[0] = sent[0].capitalize()
    except AttributeError:
        print(sent)
        
    eosPositions = [i for i,word in enumerate(sent) if word in eos and i != len(sent)-1]
    quotePositions = [i for i,word in enumerate(sent) if word == '"' and i != len(sent)-1]
    # (do not extract double quote if occurs at last position of sentence, 
    # bc nothing will follow it)

    # in recasing, must handle quote openers and quote closers separately

    # only capitalize word after endquote if word before endquote was eos
    # capitalize word after startquote ??always??

    # ex) " that's right , " he said .
    # ->  " That's right , " he said.

    # ex) I said , " yes , sir . I did . " and we started arguing .
    # ->  I said , " Yes , sir . I did . " And we started arguing .

    for j in quotePositions[::2]: # capitalize the first word inside each pair of double-quotes
        sent[j+1] = sent[j+1].capitalize()
    # makes simplifying assumption that for any sent with odd number of quotes,
    # the unpaired quote comes at the end, not the beginning
    # will be correct ~50% of the time, presumably.

    # capitalize the first word after a pair of double quotes only if
    # word before endquote was in eos 
    for j in quotePositions[1::2]: 
        if j-1 in eosPositions:
            sent[j+1] = sent[j+1].capitalize()

    # capitalize first word after eos
    for j in eosPositions:
        sent[j+1] = sent[j+1].capitalize()


    return sent







# wrapper method for adding casing back to produced targets, and
# addressing fact that moses detokenizer does not correctly handle -
# given list of str(words), produces str(sentence)
def formatPrediction(sent):
    if not sent: # its an empty list, so return an empty string
        return '' # for debug, empty sentences

    sent = naiveRecase(sent) # list of str(words)
    sent = ' '.join(sent) # str(sentence)
    if "-" in sent:
        sent = sent.replace(" - ", "-")

    return sent







# remove songs; 
# remove any pair where at least one of the sentences is over length 100
def filterSentences(src_sentences, trg_sentences):
    filtered_src_sentences = []
    filtered_trg_sentences = []

    numOverlyLong = 0 # number of sentences that were over 100 but not songs
    numSongs = 0
    pairs = list(zip(src_sentences, trg_sentences))
    print("{} pairs before filtering".format(len(pairs)))
    for pair in pairs:
        if '♫' in pair[0] or '♫' in pair[1] or '♪' in pair[0] or '♪' in pair[1]:
            numSongs += 1
        elif len(pair[0].split()) > 100 or len(pair[1].split()) > 100:
            numOverlyLong += 1
        else:
            filtered_src_sentences.append(pair[0])
            filtered_trg_sentences.append(pair[1])
    
    print("{} songs, {} overly long".format(numSongs, numOverlyLong))
    print("{} pairs after filtering".format(len(filtered_src_sentences)))

    return filtered_src_sentences, filtered_trg_sentences








def createNamesTable(path, namesFile, train_sentences): # list of str(sentence)'s
    namesTable = Counter() # names/proper nouns map to their frequency in corpus, everything else maps to 0
    lower_train_sentences = []
    print("creating " + namesFile[-2:] + " names table...")
    # get vocab counter for num capitalized occurrences of words
    capWords = Counter()
    for sent in train_sentences:
        lower_train_sent = []
        tokens = sent.split()
        for word in tokens:
            if word[0].isupper():
                capWords[word] += 1
                #print(capWords)
            lower_train_sent.append(word.lower())
        lower_train_sentences.append(lower_train_sent)

    #print(lower_train_sentences)
    allWords = Counter()
    # get vocab counter for total occurrences of words 
    # (both capitalized and lowercase)
    for sent in lower_train_sentences:
        for word in sent:
            allWords[word] += 1

    for word in capWords:
        #print(word)
        if allWords[word.lower()] == capWords[word]:
            # num times ever occurred == num times occurred in capitalized form,
            # so assume it is a name or proper noun 
            namesTable[word] = capWords[word]
        
    # want to track all words that should not be decased in namesTable,
    # so include certain acronyms like Mr.,
    # and fix bug where corpus had a few occurrences of i and i'm

    ### ???why am i not also including I've, I'd, I'll, etc., here??
    namesTable["I"] = 1
    namesTable["I'm"] = 1
    acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
    for ac in acList:
        namesTable[ac] = 1

    with open(path + namesFile, 'w') as f:
        for name, count in sorted(namesTable.items(), key=lambda item: item[1], reverse=True):
            f.write("{} {}\n".format(name, count))

    return namesTable, len(namesTable)







# determine the frequencies of words in the src and trg corpuses
def to_counters(src_sentences, trg_sentences):
    src_counter = Counter()
    for src_sentence in src_sentences:
        src_counter.update(src_sentence)

    trg_counter = Counter()
    for trg_sentence in trg_sentences:
        trg_counter.update(trg_sentence)

    print("lengths of vocabs before trimming: src: %d, trg: %d" % (len(src_counter.items()), len(trg_counter.items())))
    return src_counter, trg_counter



# keep only the top srcK most frequent words of src vocab and trgK most frequent words of trg vocab
def trim_vocabs2(src_counter, trg_counter, srcK=10000, trgK=10000):
    sorted_src_counter = sorted(src_counter.items(), key = lambda c: c[1], reverse=True)
    topKsrcwordcountpairs = sorted_src_counter[:srcK]
    topKsrcwords = [word for word, count in topKsrcwordcountpairs]
    trimmed_src_vocab = set(topKsrcwords)

    sorted_trg_counter = sorted(trg_counter.items(), key = lambda c: c[1], reverse=True)
    topKtrgwordcountpairs = sorted_trg_counter[:trgK]
    topKtrgwords = [word for word, count in topKtrgwordcountpairs]
    trimmed_trg_vocab = set(topKtrgwords)

    print("lengths of vocabs after trimming: src: %d, trg: %d" % (len(trimmed_src_vocab), len(trimmed_trg_vocab)))
    return trimmed_src_vocab, trimmed_trg_vocab



# keep only the src and trg words that occurred over the src and trg threshold frequencies, respectively
def trim_vocabs(src_counter, trg_counter, srcThres=2, trgThres=2):
    srcwords = [word for word, count in src_counter.items() if count >= srcThres]
    trimmed_src_vocab = set(srcwords)

    trgwords = [word for word, count in trg_counter.items() if count >= trgThres]
    trimmed_trg_vocab = set(trgwords)

    print("lengths of vocabs after trimming: src: %d, trg: %d" % (len(trimmed_src_vocab), len(trimmed_trg_vocab)))
    return trimmed_src_vocab, trimmed_trg_vocab



# src_sentences is a list of lists of str(word)'s
def removeOOV(sentences, trimmed_vocab):
    new_sentences = []
    for sent in sentences:
        new_sent = [token if token in trimmed_vocab else '<unk>' for token in sent]
        new_sentences.append(new_sent)
        
    return new_sentences

    





# prepend and append trg sentences with start-of-sentence token, <sos>, and end-of-sentence token, <eos>, respectively
# trg_sentences is a list of lists of str(word)'s
def add_start_end_tokens(trg_sentences):
    ###???is this list concat a constant time op??? is it creating a copy???
    return [['<sos>'] + sent + ['<eos>'] for sent in trg_sentences]
    






# construct mappings from words to indices, and vice-versa.
# these vocabs consist of the special tokens, as well as the top K words from each corpus (see trim_vocabs())
# for convenience, use diff values for tokens when working with synthetic data
# (single ch's x, p, q, and u instead of <pad>, <sos>, <eos>, and <unk>, respectively), 
# for convenience in reading the predictions
def computeVocabs(src_sentences, trg_sentences):
    srcV = {'<pad>':0, '<unk>':1}
    trgV = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3} # only trg sentences have sos and eos tokens

    for sent in src_sentences:
        for word in sent:
            if word not in srcV:
                srcV[word] = len(srcV)

    idx_to_src_word = dict((v,k) for (k,v) in srcV.items())

    for sent in trg_sentences:
        for word in sent:
            if word not in trgV:
                trgV[word] = len(trgV)
    
    idx_to_trg_word = dict((v,k) for (k,v) in trgV.items())

    return srcV, trgV, idx_to_src_word, idx_to_trg_word



# vocab files contain lines of the form "<subword> <count>"
# i assume i'm supposed to conflate the same morpheme of both langs into single entry/meaning
def computeBPEvocabs(src_vocab_file, trg_vocab_file):
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


def toIndices(sentences, vocab):
    idx_sentences = []
    for sent in sentences:
        # try:
        #     idx_sent = [vocab[word] for word in sent]
        # except KeyError:
        #     print("error: {} contains unknown word".format(sent))

        idx_sent = []
        for word in sent:
            try:
                idx_sent.append(vocab[word])
            except KeyError:
                ### not quite sure why this happens. maybe some sort of quirk in the subword-nmt script. i'll figure it out later.
                # one clue is that they all seem to be special unicode ch's. maybe i am not using proper encoding scheme.
                print("warning: unknown word: {} in sent: {}".format(word, sent))
                print("removed it from the sentence.")
                print()
        idx_sentences.append(idx_sent)

    return idx_sentences



###???do i even use this???
# assuming all parameters contain normalized sentences, creates normalized corpus counterparts in file form to be processed by learnBPE
def normalizeFiles(path, train_src_sentences, train_trg_sentences, dev_src_sentences, dev_trg_sentences, test_src_sentences):
    with open(path + "norm_train.de", "w") as f:
        for sent in train_src_sentences:
            f.write(' '.join(sent) + '\n')
   
    with open(path + "norm_train.en", "w") as f:
        for sent in train_trg_sentences:
            f.write(' '.join(sent) + '\n')
   
    with open(path + "norm_dev.de", "w") as f:
        for sent in dev_src_sentences:
            f.write(' '.join(sent) + '\n')
   
    with open(path + "norm_dev.en", "w") as f:
        for sent in dev_trg_sentences:
            f.write(' '.join(sent) + '\n')
   
    with open(path + "norm_test.de", "w") as f:
        for sent in test_src_sentences:
            f.write(' '.join(sent) + '\n')
   







if __name__=='__main__':
    
    # run from within code directory
    path = "../iwslt16_en_de/" # path to corpuses
    norm_texts = normalizeCorpuses(path)

    
    
    
    #vocab, idx_to_subword = computeBPEvocabs(path + "vocab.de", path + "vocab.en")
    
    # src_counter, trg_counter = to_counters(train_src_sentences, train_trg_sentences)

    # #trim_type = "topK"
    # trim_type = "threshold"
    # if trim_type == "threshold":
    #     trimmed_src_vocab, trimmed_trg_vocab = trim_vocabs(src_counter, trg_counter, srcThres=3, trgThres=2)
    # elif trim_type == "topK":
    #     trimmed_src_vocab, trimmed_trg_vocab = trim_vocabs2(src_counter, trg_counter, srcK=60000, trgK=45000)

    # train_src_sentences = removeOOV(train_src_sentences, trimmed_src_vocab)
    # train_trg_sentences = removeOOV(train_trg_sentences, trimmed_trg_vocab)
    # dev_src_sentences = removeOOV(dev_src_sentences, trimmed_src_vocab)

    # train_trg_sentences = add_start_end_tokens(train_trg_sentences)
    # srcV, trgV, idx_to_src_word, idx_to_trg_word = computeVocabs(train_src_sentences, train_trg_sentences)




