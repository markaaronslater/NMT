from pickle import load, dump


# utility functions for loading/storing corpuses from/to text files, examining corpus contents (for debugging), etc.

# pass arbitrary number of positional arguments. will load each of them into dict entry with their name and return the dict
# if num is not None, then will only load the first <num> lines of each corpus, starting from <_start>

# startpoint uses 1-based idxing (to match unix line numbering)
def read_corpuses(*corpus_names, path='/content/gdrive/My Drive/NMT/iwslt16_en_de/', prefix='', _start=1, num=None):
    corpuses = {}
    for corpus_name in corpus_names:
        corpuses[corpus_name] = read_corpus(corpus_name, path, prefix, _start, num)
            
    return corpuses


# read lines <start> thru <start> + <num> of corpus at text file 
def read_corpus(corpus_name, path='/content/gdrive/My Drive/NMT/iwslt16_en_de/', prefix='', _start=1, num=None):
    assert prefix in ['', 'word_', 'subword_joint_', 'subword_ind']
    with open(path + prefix + corpus_name, mode='rt', encoding='utf-8') as f:
        corpus = f.read().strip().split('\n')
        upper = num if num is not None else len(corpus)
        start = _start-1 # convert to 0-based idxing

    return corpus[start:start+upper]


# prefix can be decased_ if decased
# or bpe_ (if decased and word-split)
# or even tok_ (if just segmented and tokenized) (haven't added support for this yet)
# each corpus in corpuses is expected to be tokenized, at the very least (List[List[str]])
# write each corpus inside corpuses to a text file.
# !!!change this spec so that joins sentences prior to calling, so corpus is list of str(sentence)'s
def write_corpuses(corpuses, path='/content/gdrive/My Drive/NMT/iwslt16_en_de/', prefix='', _start=1, num=None, delim='\n'):
    for corpus_name in corpuses:
        write_corpus(corpus_name, corpuses[corpus_name], path, prefix, _start, num, delim)


# e.g., if delim='\n', sentences placed on contiguous lines, 
# or if delim='\n\n', sentences separated by blank lines (to meet stanza tokenizer spec) 
def write_corpus(corpus_name, corpus, path='/content/gdrive/My Drive/NMT/iwslt16_en_de/', prefix='', _start=1, num=None):
    assert prefix in ['', 'decased_', 'bpe_', 'tok_']
    upper = num if num is not None else len(corpus)
    start = _start-1 # convert to 0-based idxing
    with open(path + prefix + corpus_name, mode='wt', encoding='utf-8') as f:
        for sent in corpus[start:start+upper]:
            f.write(' '.join(sent))
            f.write('\n')


# wrapper function that reads and white-space splits a pre-tokenized corpus stored in a file.
def read_tokenized_corpuses(*corpus_names, path='/content/gdrive/My Drive/NMT/iwslt16_en_de/', prefix=''):
    corpuses = read_corpuses(*corpus_names, path, prefix)
    tokenize_corpuses(corpuses)
    ref_corpuses = get_references(corpuses) # for estimating model quality after each epoch using corpus_bleu
    
    return corpuses, ref_corpuses


def tokenize_corpuses(corpuses):
    for corpus_name in corpuses:
        tokenize_corpus(corpuses[corpus_name])


def tokenize_corpus(corpus):
    for i, sent in enumerate(corpus):
        corpus[i] = corpus[i].split()





# -input: corpuses is List[List[str]]
# -output: ref_corpuses, which is List[List[List[str]]],
# where middle list is a singleton.
# (bc I only ever provide a single reference translation for
# any given source sentence). 
def get_references(corpuses, num_overfit=10):
    ref_corpuses = {}
    # for debugging/overfitting to first <num_overfit> sentences of trainset:
    ref_corpuses["train.en"] = [[target_sent] for target_sent in corpuses["train.en"][:num_overfit]]

    # for actual dev set:
    ref_corpuses["dev.en"] = [[target_sent] for target_sent in corpuses["dev.en"]]
    
    return ref_corpuses


# return true if is a source corpus, and false if is a target corpus
def is_src_corpus(corpus_name, src_corpus_suffix="de"):
    return corpus_name[-2:] == src_corpus_suffix


# print out first <num> sentences of each corpus
def print_corpuses(corpuses, num=5):
    for corpus_name in corpuses:
        print(corpus_name)
        print_corpus(corpuses[corpus_name], num)
    

def print_corpus(corpus, num=5):
    for sent in corpus[:num]:
        print(sent)
    print()


# prints the morphological data associated with each word of each sentence.
# designed for printing output of stanza processors.
def print_processed_corpuses(corpuses, num=5):
    for corpus_name in corpuses:
        print(corpus_name)
        print_processed_corpus(corpuses[corpus_name], num)


# each corpus is a stanza Document object.
def print_processed_corpus(doc, num=5):
    for i, sent in enumerate(doc.sentences[:num]):
        print(f"####### sentence {i+1}: #######")
        for word in sent.words: 
            print(f'word: {word.text}\t\tupos: {word.upos}\txpos: {word.xpos}')
        print("###############################")
    print()