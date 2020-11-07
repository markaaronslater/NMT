from pickle import load

from src.preprocessing.corpus_utils import is_src_corpus

def truecase_corpuses(*corpus_names, corpus_path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/'):
    stanza_path = corpus_path + 'stanza_outputs/'
    truecased_path = corpus_path + 'truecased/'
    num_corpus_pieces = load(open(f"{stanza_path}num_corpus_pieces.pkl", 'rb'))
    for corpus_name in corpus_names:
        should_lower = should_lower_de if is_src_corpus(corpus_name) else should_lower_en

        # each processed corpus is a list of Stanza Sentence objects.
        open(f"{truecased_path}word_{corpus_name}", 'w').close() # clear existing file (appended to inside truecase)
        num_pieces = num_corpus_pieces[corpus_name]
        # entire processed corpus does not fit in memory. apply truecasing in pieces.
        for piece_number in range(1, num_pieces+1):
            corpus_piece = load(open(f"{stanza_path}stanza_{corpus_name}_{piece_number}.pkl", 'rb'))
            print(f"truecasing piece {piece_number} of {corpus_name}...")
            truecase_corpus(corpus_name, truecased_path, corpus_piece, should_lower)  
    
    print("done.")
    

# -before calling, corpuses is dict that maps each corpus name to a
# stanza Document object corresponding to its entire stanza-processed corpus.
# -after calling, corpuses is a dict that maps each corpus name to a list
# of sentences in the corpus (where sentence is list of words),
# where each sentence has been decased using linguistic heuristics that
# leverage morphological data supplied by pos-tagger.
# (this is more accurate than, e.g., the Moses truecaser).
def truecase_corpus(corpus_name, truecased_path, sentences, should_lower):
    # append all corpus pieces into single truecased corpus text file.
    with open(f"{truecased_path}word_{corpus_name}", mode='a', encoding='utf-8') as f:
        for sent in sentences:
            truecase_sentence(sent, should_lower)
            f.write(' '.join([word.text for word in sent.words]))
            f.write('\n')


def truecase_sentence(sent, should_lower):
    for j, word in enumerate(sent.words):
        if j == 0 and should_lower(word) or should_lower(word, sent.words[j-1]):
            sent.words[j].text = word.text.lower()



### the following should_be_lower() functions exploit domain knowledge about
# the given language for designing heuristics about whether or not to decase
# a given word. given a word, returns true if it should be decased.
eos_symbols = {".", "!", "?", "\"", ":", "..", "...", "...."}
# serve as "capitalization prefixes"
# -> a token whose subsequent word ought to be capitalized for syntactic reason.

# TODO: find better heuristic than this for distinguishing 'she/it' from 'they' from 'You':
# -if 'Sie' is at cap_location, it could either be 'she/it/they/You',
# where 'You' is formal. stanfordnlp pos tagger trained on corpus that
# uses automated labeling of number, plural, and gender, so treats every
# instance of 'Sie' as if it were 3rd person plural, with no gender.
# -therefore, cannot use that info to distinguish 'You' (which should not
# be lower cased) from 'they', (which should be lower cased).
# -however, if means 'she/it', then will be singular, feminine, so could use
# that info to at least properly decase those senses.

# -word and previous_word are each stanza Word objects.
# -word is the current word we are deciding if should be decased or not.
# -previous_word is None when <word> is first of the sentence.
def should_lower_de(word, previous_word=None):
    if not previous_word or previous_word.text in eos_symbols:
        # word is first word of sentence, or previous word is a "capitalization prefix"
        if word.upos == 'PRON':
            if word.text != 'Sie' or (word.text == 'Sie' and "Gender=Fem" in word.feats):
                return True
        elif word.upos != 'PROPN' and word.upos != 'NOUN':
            return True

    return False


# decase all words at cap_locations unless they are proper nouns, or are the pronoun 'I'.
# (sentence is already tokenized, so "I" conjunctions, e.g., "I've", appear as "I 've")
def should_lower_en(word, previous_word=None):
    if not previous_word or previous_word.text in eos_symbols:
        if word.upos != 'PROPN' and word.text != 'I':
            return True

    return False