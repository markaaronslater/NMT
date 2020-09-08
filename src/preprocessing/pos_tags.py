#import stanza
from pickle import load, dump
from math import ceil

#corpus_types = {""}
# pass arbitrary number of positional arguments. will load each of them into dict entry with their name and return the dict
# if num is not None, then will only load the first num lines of each corpus
def load_docs3(path, *corpus_names, num=None):
    corpuses = {}
    for corpus_name in corpus_names:
        with open(path + corpus_name, mode='rt', encoding='utf-8') as f:
            corpuses[corpus_name] = f.read().strip().split('\n')
            if num is not None:
                # only keep the first <num> sentences of the corpus
                corpuses[corpus_name] = corpuses[corpus_name][:num]
            
    return corpuses


# returns dict where each corpus_name maps to its number of sentences
def corpus_lengths(corpuses):
    return {corpus_name:len(corpuses[corpus_name]) for corpus_name in corpuses}



# tags, tokenizes, segments corpuses.
# default piece_size is large enough that dev and test sets all fit inside single piece, but train sets will each get split into roughly 20 pieces.
# corpus is a list of len(corpus) Document objects
def apply_stanfordnlp_processor(corpus_name, corpus, processor, path='/content/gdrive/My Drive/iwslt16_en_de/', piece_size=10000):
    for j in range(0, len(corpus), piece_size):
        corpus_piece = corpus[j:j+piece_size] # list of piece_size Document objects
        piece_number = j // piece_size + 1
        file_name = f"{path}processed_{corpus_name}_{piece_number}"
        print(f"processing piece {piece_number} of {corpus_name}...")
        docs = [] # List[Document]
        for sentence in corpus_piece:
            docs.append(processor(sentence))

        store_corpus(docs, file_name)




def process_corpuses(corpuses, src_processor, trg_processor, path='/content/gdrive/My Drive/iwslt16_en_de/', piece_size=10000):

    # first, store the number of pieces that will be associated with each corpus
    num_corpus_pieces = {corpus_name:ceil(len(corpuses[corpus_name]) / piece_size) for corpus_name in corpuses}
    print(f"pieces: {num_corpus_pieces}\n")
    dump(num_corpus_pieces, open(path + 'num_corpus_pieces', 'wb'))

    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            apply_stanfordnlp_processor(corpus_name, corpuses[corpus_name], src_processor, path, piece_size)
        else:
            apply_stanfordnlp_processor(corpus_name, corpuses[corpus_name], trg_processor, path, piece_size)





# pass it a single corpus, rather than dict of all corpuses.
# prints tag info
def print_processed_corpus(corpus, num=5):
    for doc in corpus[:num]:
        for sent in doc.sentences:
            for word in sent.words: 
                print(f'word: {word.text}\t\tupos: {word.upos}\txpos: {word.xpos}')
            print("########################################")
        print('\n\n')
    

def print_processed_corpuses(corpuses, num=5):
    for corpus_name in corpuses:
        print_processed_corpus(corpuses[corpus_name], num)


# return true if is a source corpus, and false if is a target corpus
def is_src_corpus(corpus_name, src_corpus_suffix="de"):
    return corpus_name[-2:] == src_corpus_suffix




# pass absolute path to filename to be saved to
# corpus is a list of objects
def store_corpus(corpus, path):
    dump(corpus, open(path, 'wb'))
    print(f"saved to {path}")
    print()

def load_corpus(path):
    return load(open(path, 'rb'))




def merge_corpus_pieces(corpus_name, num_pieces, path='/content/gdrive/My Drive/iwslt16_en_de/'):
    merged_corpus = []
    for i in range(1, num_pieces+1):
        merged_corpus += load_corpus(f"{path}processed_{corpus_name}_{i}")

    return merged_corpus





def get_processed_corpuses(*corpuses, path='/content/gdrive/My Drive/iwslt16_en_de/'):
    # get the number of pieces that each corpus is distributed across
    num_corpus_pieces = load(open(path + 'num_corpus_pieces', 'rb'))
    print(f"pieces: {num_corpus_pieces}\n")
    processed_corpuses = {}
    for corpus_name in corpuses:
        processed_corpuses[corpus_name] = merge_corpus_pieces(corpus_name, num_corpus_pieces[corpus_name], path)

    return processed_corpuses





















# each sentence of the corpus corresponds to a Document object in stanza.
# Document object consists of one or more Sentence objects. (in case a src or trg sentence of the corpus is actually more than one sentence)

def decase_corpuses(corpuses, num=5):
    decased_corpuses = {}
    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            decased_corpuses[corpus_name] = german_decase(corpuses[corpus_name], num)  
        else:
            decased_corpuses[corpus_name] = english_decase(corpuses[corpus_name], num)
    

    return decased_corpuses



### NOTE: the following decase functions are tailored toward specific languages. when add support for translating other languages, will write specific decase functions for each of them
### TODO: convert to closures, and have specific function "should_be_decased() that call rather than directly performing english or german specific check"
# decase all words at cap_locations unless they are proper nouns, or are the pronoun 'I'.

# after this stage, the pos and morphological data is no longer needed, so just returns the tokenized, decased corpus as List[List[Str]]
def english_decase(corpus, num=5):
    decased_sentences = []
    for doc in corpus[:num]:
        # build single sentence out of the potentially multiple sentences in the line
        decased_sent = [] # List[Word]
        for sent in doc.sentences:
            # cap_locations (other than beginning of sentence) include the sent position immediately after a double-quote or a colon (which I refer to as 'cap_location prefixes')
            cap_prefixes = ['"', ':']
            for i, word in enumerate(sent.words):
                if i == 0:
                    first_word = sent.words[0]
                    if first_word.upos != 'PROPN' and first_word.text != 'I':
                        sent.words[0].text = first_word.text.lower()
                else:
                    prev_word = sent.words[i-1]
                    if prev_word.text in cap_prefixes:
                        if word.upos != 'PROPN' and word.text != 'I':
                            sent.words[i].text = word.text.lower()

            decased_sent += sent.words
        # only want to keep the text fields       
        decased_sentences.append([word.text for word in decased_sent])

    return decased_sentences



# decase all words at cap_locations unless they are proper nouns or common nouns, or are the pronoun 'I'.
# TODO: find better heuristic than this for distinguishing 'she/it' from 'they' from 'You':
# if 'Sie' is at cap_location, it could either be 'she/it/they/You', where 'You' is formal. stanfordnlp pos tagger trained on corpus that uses automated labeling of number, plural, and gender, so treats every instance of 'Sie' as if it were 3rd person plural, with no gender. therefore, cannot use that info to distinguish 'You' (which should not be lower cased) from 'they', (which should be lower cased). however, if means 'she/it', then will be singular, feminine, so could use that info to at least properly decase those senses.
def german_decase(corpus, num=5):
    decased_sentences = []
    for doc in corpus[:num]:
        decased_sent = [] # List[Word]
        for sent in doc.sentences:
            cap_prefixes = ['"', ':']
            for i, word in enumerate(sent.words):
                if i == 0:
                    # first word always at a cap_location, so handle separately
                    word = sent.words[0]
                    if word.upos == 'PRON':
                        if word.text != 'Sie' or (word.text == 'Sie' and "Gender=Fem" in word.feats):
                            sent.words[0].text = word.text.lower()
                    elif word.upos != 'PROPN' and word.upos != 'NOUN':
                        sent.words[0].text = word.text.lower()
                else:
                    prev_word = sent.words[i-1]
                    if prev_word.text in cap_prefixes:
                        if word.upos == 'PRON':
                            if word.text != 'Sie' or (word.text == 'Sie' and "Gender=Fem" in word.feats):
                                sent.words[i].text = word.text.lower()
                        elif word.upos != 'PROPN' and word.upos != 'NOUN':
                            sent.words[i].text = word.text.lower()
                    

            decased_sent += sent.words
        # only want to keep the text fields       
        decased_sentences.append([word.text for word in decased_sent])

    return decased_sentences


def test_german_decase(src_processor):
    german_decase([src_processor('Sie haben etwas namens RNA.')], num=1)




# corpuses is List[List[str]]
# returns ref_corpuses, which is List[List[List[str]]], where middle list is a singleton, bc only i only ever provide a single reference translation for any given source sentence. This conforms to nltk corpus_bleu fn
def get_references(corpuses):
    ref_corpuses = {}
    # for debugging/overfitting to small fraction of trainset:
    ref_corpuses["train.en"] = [[target_sent] for target_sent in corpuses["train.en"]]

    # for actual dev set:
    ref_corpuses["dev.en"] = [[target_sent] for target_sent in corpuses["dev.en"]]

    return ref_corpuses
    
# get the references
# if use bpe, store in bpe_ files
# filter out the songs (should i even bother?)