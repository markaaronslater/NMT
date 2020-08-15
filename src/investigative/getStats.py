from processCorpuses import load_docs, to_sentences, filterSentences, normalizeCorpuses, createNamesTable
from collections import Counter, defaultdict
import re
import statistics
from unicodedata import normalize
from pickle import load, dump, HIGHEST_PROTOCOL

# compute histogram of sentence lengths of src and trg corpuses, in order to determine appropriate universal pad size,
# mask ranges, etc.
def histoLengths(src_sentences, trg_sentences):
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


def getStats(src_sentences, trg_sentences):
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





# produce corpus counterparts with line-numbering, to ease corresponding prediction look-ups, etc.
def numberCorpuses(path, train_src_sentences, train_trg_sentences, dev_src_sentences, dev_trg_sentences, test_src_sentences, filtered=True):
    prefix = "num_"
    if filtered:
        train_prefix = "numf_" # only makes a difference for the training sets
    else:
        train_prefix = prefix
    with open(path + train_prefix + "train.de", "w") as f:
        # use 1-based indexing
        for i, sent in enumerate(train_src_sentences):
            f.write(str(i+1) + ': ' + sent + '\n')
   
    with open(path + train_prefix + "train.en", "w") as f:
        for i, sent in enumerate(train_trg_sentences):
            f.write(str(i+1) + ': ' + sent + '\n')
   
    with open(path + prefix + "dev.de", "w") as f:
        for i, sent in enumerate(dev_src_sentences):
            f.write(str(i+1) + ': ' + sent + '\n')
   
    with open(path + prefix + "dev.en", "w") as f:
        for i, sent in enumerate(dev_trg_sentences):
            f.write(str(i+1) + ': ' + sent + '\n')
   
    with open(path + prefix + "test.de", "w") as f:
        for i, sent in enumerate(test_src_sentences):
            f.write(str(i+1) + ': ' + sent + '\n')


# acronyms containing at least one period
def getRegex(train_sentences):
    #acronyms = re.compile(r'\w+\.\w+[.\w]*')
    #acronyms = re.compile(r'\b\w\w\.\s') 
    acronyms = re.compile(r'\W\'\w+|\w+\'\W') 


    for sent in train_sentences:
        acs = acronyms.findall(sent)
        if len(acs) > 0:
            print(acs)



def geteosinfo(sentences):
    eosSymbols = ['.','?','!', '"', '♫', ']', ':', '♪', ',']
    numSent = 0 # num sentences not ending in .
    for sent in sentences:
        if sent[-1] not in eosSymbols:
            print(sent)
            numSent += 1
    print(numSent)

def getspecialCharinfo(sentences):
    numSent = 0 # num sentences not ending in .
    
    for i, sent in enumerate(sentences):
        new_sent = normalize('NFD', sent).encode('ascii', 'ignore')
        new_sent = new_sent.decode('UTF-8')
        if sent != new_sent:
            print("{} {}".format(i, sent))
            print(new_sent)
            print()
            numSent += 1
    print(numSent)


def getsongs(sentences):
    numSent = 0 # num sentences not ending in .
    
    for i, sent in enumerate(sentences):
        if '♫' in sent or '♪' in sent:
            print("{} {}".format(i, sent))
            print()
            numSent += 1
    print(numSent)


def multiplePeriods(sentences):
    numSent = 0 
    
    for i, sent in enumerate(sentences):
        #sent = ' '.join(sent)
        if sent.count('.') > 1:
            
            print("{} {}".format(i, sent))
            print()
            numSent += 1
    print(numSent)



def multipleQuotes(sentences):
    numSent = 0 
    
    for i, sent in enumerate(sentences):
        if sent.count('"') > 0:
            
            print("{} {}".format(i, sent))
            print()
            numSent += 1
    print(numSent)





# must pass corpuses in expected order
def naiveCorpusSegmenter(corpuses):
    acronyms = re.compile(r'\w+\.\w+[.\w]*|Mr\.|Mrs\.|Ms\.|Dr\.|St\.|Mt\.|Lt\.|lbs?\.|sq\.|ft\.') 
    ellipses = re.compile(r'(\.\.\.*)')
    corpus_names = ["train.de", "train.en", "dev.de", "dev.en", "test.de"]
    for corpus_idx, corpus in enumerate(corpuses):
        corpus_name = corpus_names[corpus_idx]
        with open(path + "seg_" + corpus_name, "w") as f:
            print("segmenting " + corpus_name + "...")
            #num = 0
            for sent in corpus:
                # overwrite sentence such that each ellipse surrounded by whitespace
                s = re.sub(ellipses, r' \1 ', sent)

                # get locations of new string spanned by the ellipses
                ellipseSpans = []
                for m in ellipses.finditer(s):
                    ellipseSpans += list(range(m.start(), m.end())) # flatten list
                    
                # get locations of the new string spanned by the acronyms
                acronymSpans = []
                for m in acronyms.finditer(s):
                    acronymSpans += list(range(m.start(), m.end())) 

                # remove them from consideration when segmenting periods of sentence
                possiblePeriods = set(range(len(s))) - set(acronymSpans) - set(ellipseSpans)
                realPeriods = [i for i in possiblePeriods if s[i] == '.']

                if len(realPeriods) > 0:
                    end = realPeriods[0]
                    segmented_sent = s[:end]
                    start = end
                    for end in realPeriods[1:]:
                        # add space to the left of each real period: ex) hey. -> hey .
                        segmented_sent = segmented_sent + ' ' + s[start:end]
                        start = end
                    segmented_sent = segmented_sent + ' ' + s[end]
                else:
                    segmented_sent = s

                # if corpus_name == "train.en":
                #     if len(realPeriods) > 1:
                #         print(segmented_sent)
                #         num += 1
                f.write(segmented_sent + '\n')
            #print("{} lines contain more than one sentence".format(num))
            #print()





if __name__=='__main__':
    # run from within code directory
    path = "../iwslt16_en_de/" # path to corpuses
    # texts = load_docs(path)
    # texts = to_sentences(texts)    
    # train_src_sentences, train_trg_sentences, dev_src_sentences, dev_trg_sentences, test_src_sentences = texts[0], texts[1], texts[2], texts[3], texts[4]
    # train_src_sentences, train_trg_sentences = texts[0], texts[1]
    # train_src_sentences, train_trg_sentences = filterSentences(train_src_sentences, train_trg_sentences)
    # texts[0], texts[1] = train_src_sentences, train_trg_sentences

    # normalizeCorpuses(texts, path, decase=False) #do not decase when construct namesTable

  
    #numberCorpuses(path, train_src_sentences, train_trg_sentences, dev_src_sentences, dev_trg_sentences, test_src_sentences, filtered=True)
    #multiplePeriods(dev_trg_sentences)
    #multipleQuotes(dev_trg_sentences)

    norm_ref_texts = load_docs(path, ref=True)
    norm_ref_texts = to_sentences(norm_ref_texts)
    train_src_sentences, train_trg_sentences, dev_src_sentences, dev_trg_sentences, test_src_sentences = norm_ref_texts[0], norm_ref_texts[1], norm_ref_texts[2], norm_ref_texts[3], norm_ref_texts[4]


    de_namesTable, de_count = createNamesTable(path, "names.de", train_src_sentences)
    # store the dict as in a pickle file
    # with open(path + "names.de", 'wb') as f:
    #     dump(de_namesTable, f)

    en_namesTable, en_count = createNamesTable(path, "names.en", train_trg_sentences)
    # with open(path + "names.en", 'wb') as f:
    #     dump(en_namesTable, f)
    print("de names count: {}".format(de_count))
    print("en names count: {}".format(en_count))

    # still need to use Iwords, bc i and i'm occurred a few times in the trainset, etc.