from corpus_utils import write_corpuses, is_src_corpus
from apply_stanza_processors import retrieve_stanza_outputs

def truecase_corpuses(*corpus_names, path='/content/gdrive/My Drive/iwslt16_en_de/'):
    corpuses = retrieve_stanza_outputs(corpus_names, path) ### doing inside here, rather than wrapper fn
    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            truecase(corpus_name, corpuses, should_lower_de, path)  
        else:
            truecase(corpus_name, corpuses, should_lower_en, path)

    # decased corpuses directly used by model employing vanilla word-level vocab.
    # (other vocab-types process the corpuses even further).
    write_corpuses(corpuses, path, 'word_') 


# def decase_corpuses(corpuses, path='/content/gdrive/My Drive/iwslt16_en_de/'):
#     for corpus_name in corpuses:
#         if is_src_corpus(corpus_name):
#             decase(corpus_name, corpuses, should_lower_de, path)  
#         else:
#             decase(corpus_name, corpuses, should_lower_en, path)

#     # decased corpuses directly used by model employing vanilla word-level vocab.
#     # (other vocab-types process the corpuses even further).
#     write_corpuses(corpuses, path, 'word_') 

# -before calling, corpuses is dict that maps each corpus name to a
# stanza Document object corresponding to its entire stanza-processed corpus.
# -after calling, corpuses is a dict that maps each corpus name to a list
# of sentences in the corpus (where sentence is list of words),
# where each sentence has been decased using linguistic heuristics that
# leverage morphological data supplied by pos-tagger.
def truecase(corpus_name, corpuses, should_lower, path='/content/gdrive/My Drive/iwslt16_en_de/', _start=1, num=None):
    doc = corpuses[corpus_name]
    decased_corpus = []
    for i, sent in enumerate(doc.sentences):
        for j, word in enumerate(sent.words):
            if j == 0 and should_lower(word) or should_lower(word, sent.words[j-1]):
                sent.words[j].text = word.text.lower()

        decased_corpus.append([word.text for word in sent.words])

    corpuses[corpus_name] = decased_corpus


### the following should_be_lower() functions exploit domain knowledge about the given language for designing heuristics about whether or not to decase a given word. given a word, returns true if it should be decased.
eos_symbols = {".", "!", "?", "\"", ":", "..", "...", "...."}
# serve as "capitalization prefixes"
# -> a token whose subsequent word ought to be capitalized for syntactic reason.

# TODO: find better heuristic than this for distinguishing 'she/it' from 'they' from 'You':
# if 'Sie' is at cap_location, it could either be 'she/it/they/You',
# where 'You' is formal. stanfordnlp pos tagger trained on corpus that
# uses automated labeling of number, plural, and gender, so treats every
# instance of 'Sie' as if it were 3rd person plural, with no gender.
# therefore, cannot use that info to distinguish 'You' (which should not
# be lower cased) from 'they', (which should be lower cased).
# however, if means 'she/it', then will be singular, feminine, so could use
# that info to at least properly decase those senses.

# -word and previous_word are each stanza Word objects.
# -word is the current word we are deciding if should be decased or not.
# -previous_word is None when <word> is first of the sentence.
def should_lower_de(word, previous_word=None):
    if not previous_word or previous_word in eos_symbols:
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
    if not previous_word or previous_word in eos_symbols:
        if word.upos != 'PROPN' and word.text != 'I':
            return True

    return False