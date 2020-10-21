from corpus import write_corpus, is_src_corpus

# each sentence of the corpus corresponds to a Document object in stanza.
# Document object consists of one or more Sentence objects. (in case a src or trg sentence of the corpus is actually more than one sentence)
def decase_corpuses(corpuses, path='/content/gdrive/My Drive/iwslt16_en_de/', num=5):
    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            decase(corpus_name, corpuses, should_be_lower_de, path, num)  
        else:
            decase(corpus_name, corpuses, should_be_lower_en, path, num)


# each corpus in corpuses is a list of Document objects (under Stanza spec), bc working with preprocess_phase1() outputs.
# after this stage, the pos and other morphological data are no longer needed, so just overwrites corpuses to hold the decased corpus as List[List[Str]], and writes it to a file for later use by jointBPE.sh script, if using subword vocab.
# should_be_lower is a function.


# TODO: with new preprocess_phase1, the Document object is the entire corpus, so get each sentence via 'for sent in doc.sentences'.
# now, add all other eos punctuation to cap_prefixes, since no longer labeling sentences...
def decase(corpus_name, corpuses, should_be_lower, path='/content/gdrive/My Drive/iwslt16_en_de/', num=None):
    upper = num if num is not None else len(corpuses[corpus_name])
    for i, doc in enumerate(corpuses[corpus_name][:upper]):
        decased_sent = [] # List[Word]
        for sent in doc.sentences:
            for j, word in enumerate(sent.words):
                if j == 0 and should_be_lower(word) or should_be_lower(word, sent.words[j-1]):
                    sent.words[j].text = word.text.lower()

            decased_sent += sent.words # merge all segmented sentences of the line into single sentence

        corpuses[corpus_name][i] = [word.text for word in decased_sent]




### the following should_be_lower() functions exploit domain knowledge about the given language for designing heuristics about whether or not to decase a given word. given a word, returns true if it should be decased.


# TODO: find better heuristic than this for distinguishing 'she/it' from 'they' from 'You':
# if 'Sie' is at cap_location, it could either be 'she/it/they/You', where 'You' is formal. stanfordnlp pos tagger trained on corpus that uses automated labeling of number, plural, and gender, so treats every instance of 'Sie' as if it were 3rd person plural, with no gender. therefore, cannot use that info to distinguish 'You' (which should not be lower cased) from 'they', (which should be lower cased). however, if means 'she/it', then will be singular, feminine, so could use that info to at least properly decase those senses.
# word and previous_word are each Word objects under Stanza spec
# previous_word is None when current_word is first of the sentence.
# word is the current word we are deciding if should be decased or not.
def should_be_lower_de(word, previous_word=None):
    if not previous_word or previous_word in ['"', ':']:
        # word is first word of sentence, or previous word is a "capitalization prefix" (a word whose subsequent word ought to be capitalized for syntactic reason)
        if word.upos == 'PRON':
            if word.text != 'Sie' or (word.text == 'Sie' and "Gender=Fem" in word.feats):
                return True
        elif word.upos != 'PROPN' and word.upos != 'NOUN':
            return True

    return False


# decase all words at cap_locations unless they are proper nouns, or are the pronoun 'I'.
# (tokenized at this point, so I conjunctions, e.g., I've, appear as I 've)
def should_be_lower_en(word, previous_word=None):
    if not previous_word or previous_word in ['"', ':']:
        if word.upos != 'PROPN' and word.text != 'I':
            return True

    return False
        

# def test_german_decase(src_processor):
#     german_decase([src_processor('Sie haben etwas namens RNA.')], num=1)