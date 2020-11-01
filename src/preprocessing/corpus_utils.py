from pickle import load, dump

# utility functions for loading/storing corpuses from/to text files,
# examining corpus contents, etc.


def read_corpuses(*corpus_names, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', prefix='', _start=1, num=None):
    corpuses = {}
    for corpus_name in corpus_names:
        corpuses[corpus_name] = read_corpus(corpus_name, path, prefix, _start, num)
            
    return corpuses


# read lines <_start> thru <start> + <num> of corpus at text file 
# (<_start> uses 1-based idxing to match unix line numbering)
def read_corpus(corpus_name, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', prefix='', _start=1, num=None):
    assert prefix in ['', 'word_', 'subword_joint_', 'subword_ind_']
    with open(path + prefix + corpus_name, mode='rt', encoding='utf-8') as f:
        corpus = f.read().strip().split('\n')
        upper = num if num is not None else len(corpus)
        start = _start-1 # convert to 0-based idxing

    return corpus[start:start+upper]


# used to create "checkpoints" in the preprocessing pipeline, where rather
# than performing step again, can directly load preprocessed corpus.
def write_corpuses(corpuses, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', prefix='', _start=1, num=None):
    for corpus_name in corpuses:
        write_corpus(corpus_name, corpuses[corpus_name], path, prefix, _start, num)


def write_corpus(corpus_name, corpus, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', prefix='', _start=1, num=None):
    assert prefix in ['', 'word_', 'subword_joint_', 'subword_ind_']
    upper = num if num is not None else len(corpus)
    start = _start-1 # convert to 0-based idxing
    with open(path + prefix + corpus_name, mode='wt', encoding='utf-8') as f:
        for sent in corpus[start:start+upper]:
            f.write(' '.join(sent))
            f.write('\n')


# reads and white-space splits a pre-tokenized corpus stored in a file.
def read_tokenized_corpuses(*corpus_names, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', prefix=''):
    corpuses = read_corpuses(*corpus_names, path=path, prefix=prefix)
    tokenize_corpuses(corpuses)
    
    return corpuses


def tokenize_corpuses(corpuses):
    for corpus_name in corpuses:
        tokenize_corpus(corpuses[corpus_name])


def tokenize_corpus(corpus):
    for i, sent in enumerate(corpus):
        corpus[i] = corpus[i].split()


# load target corpuses in format required by sacreBLEU, for evaluation of predictions.
def get_references(path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/', overfit=False):
    ref_corpuses = {}
    if not overfit:
        # only one set of references, so construct singleton list of lists of sentences.
        ref_corpuses["dev.en"] = [read_corpus("dev.en", path=path)]
        #ref_corpuses["test.en"] = [read_corpus("test.en", path=path)]
    else:
        ref_corpuses["train.en"] = [read_corpus("train.en", path=path, num=10)]

    return ref_corpuses


# return true if is a source corpus, and false if is a target corpus
def is_src_corpus(corpus_name, src_corpus_suffix="de"):
    return corpus_name[-2:] == src_corpus_suffix


# print out first <num> sentences of each corpus
def print_corpuses(corpuses, num=None):
    for corpus_name in corpuses:
        print(corpus_name)
        print_corpus(corpuses[corpus_name], num)
    

def print_corpus(corpus, num=None):
    upper = num if num is not None else len(corpus)
    for sent in corpus[:upper]:
        print(sent)
    print()


# prints the morphological data associated with each word of each sentence.
# designed for printing output of stanza processors.
def print_processed_corpuses(corpuses, num=None):
    for corpus_name in corpuses:
        print(corpus_name)
        print_processed_corpus(corpuses[corpus_name], num)


# each corpus is a stanza Document object.
def print_processed_corpus(doc, num=None):
    upper = num if num is not None else len(doc.sentences)
    for i, sent in enumerate(doc.sentences[:upper]):
        print(f"####### sentence {i+1}: #######")
        for word in sent.words: 
            print(f'word: {word.text}\t\tupos: {word.upos}\txpos: {word.xpos}')
        print("###############################")
    print()
    print()