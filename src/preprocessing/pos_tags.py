#import stanza
from pickle import load, dump

# get pos tags for each word of the corpuses
# each corpus is a list of str(sentence)'s
# afterward, each str(sentence) is replaced by a Document object corresponding to that sentence


src_corpuses = {"train.de":'', "dev.de":'', "test.de":''}
trg_corpuses = {"train.en":'', "dev.en":''}
all_corpuses = {"src":src_corpuses, "trg":trg_corpuses}


def load_docs2(path, all_corpuses):
    for corpus_name in all_corpuses["src"]:
        with open(path + corpus_name, mode='rt', encoding='utf-8') as f:
            all_corpuses["src"][corpus_name] = f.read().strip()
            if all_corpuses["src"][corpus_name]: 
                all_corpuses["src"][corpus_name] = all_corpuses["src"][corpus_name].split('\n')
            else:
                all_corpuses["src"][corpus_name] = [] # f was empty (used for debugging)

    for corpus_name in all_corpuses["trg"]:
        with open(path + corpus_name, mode='rt', encoding='utf-8') as f:
            all_corpuses["trg"][corpus_name] = f.read().strip()
            if all_corpuses["trg"][corpus_name]: 
                all_corpuses["trg"][corpus_name] = all_corpuses["trg"][corpus_name].split('\n')
            else:
                all_corpuses["trg"][corpus_name] = [] # f was empty (used for debugging)
    

# map a source sentence to its line number in its corpus (1-based idxing).
# map a target sentence to its line number in its corpus.
# vice versa

# map a source sentence to its corresponding target sentence.
# vice versa
def create_lookup_tables(all_corpuses):
    sent_to_line = {"train.en":{}, "dev.en":{}, "train.de":{}, "dev.de":{}}
    line_to_sent = {"train.en":{}, "dev.en":{}, "train.de":{}, "dev.de":{}}

    for i, (src_sentence, trg_sentence) in enumerate(zip(all_corpuses["src"]["train.de"], all_corpuses["trg"]["train.en"])):
        sent_to_line["train.de"][src_sentence] = i+1 # 1-based indexing
        sent_to_line["train.en"][trg_sentence] = i+1 # 1-based indexing

        line_to_sent["train.de"][i+1] = src_sentence
        line_to_sent["train.en"][i+1] = trg_sentence

    for i, (src_sentence, trg_sentence) in enumerate(zip(all_corpuses["src"]["dev.de"], all_corpuses["trg"]["dev.en"])):
        sent_to_line["dev.de"][src_sentence] = i+1 # 1-based indexing
        sent_to_line["dev.en"][trg_sentence] = i+1 # 1-based indexing

        line_to_sent["dev.de"][i+1] = src_sentence
        line_to_sent["dev.en"][i+1] = trg_sentence

    return  {   "sent_to_line":sent_to_line, 
                "line_to_sent":line_to_sent
            }


#def print_lookup_tables(lookup_tables, start=1, num=5):
    


# returns lines <start> thru <start+num>, inclusive, of corpus <name>
def get_sentences(lookup_tables, name, line_nums=[1,2,3,4,5]):
    return [lookup_tables["line_to_sent"][name][line_num] for line_num in line_nums]


# returns line number, inside <name>, of each sentence in list <sentences>
def get_line_nums(lookup_tables, name, sentences):
    return [lookup_tables["sent_to_line"][name][sent] for sent in sentences]


# for each sentence in <sentences>, of corpus <name>, lookup the corresponding sentence in the opposite corpus
def get_partners(lookup_tables, name, sentences):
    # name of corresponding corpus, e.g., dev.de and dev.en are opposites
    opp_name = name[:-2] + "de" if name[-2:] == "en" else name[:-2] + "en" 
    line_nums = get_line_nums(lookup_tables, name, sentences)
    return get_sentences(lookup_tables, opp_name, line_nums)



# apply to each corpus of a given language (e.g., call twice: for src lang and trg lang)
# tags, tokenizes, segments corpuses
def apply_stanfordnlp_processor(corpuses, processor, num=5):
    #nlp_en = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
    for corpus_name in corpuses:
        corpus = corpuses[corpus_name]
        for i, sentence in enumerate(corpus[:num]):
            corpus[i] = processor(sentence) # corpus[i] is now a Document object


def process_corpuses(all_corpuses, src_processor, trg_processor, num=5):
    apply_stanfordnlp_processor(all_corpuses["src"], src_processor, num)
    apply_stanfordnlp_processor(all_corpuses["trg"], trg_processor, num)


# print str(sentence)'s
def print_corpuses2(all_corpuses, num=5):
    for corpus_name in all_corpuses["src"]:
        corpus = all_corpuses["src"][corpus_name]
        for sent in corpus[:num]:
            print(sent)
        print()

    for corpus_name in all_corpuses["trg"]:
        corpus = all_corpuses["trg"][corpus_name]
        for sent in corpus[:num]:
            print(sent)
        print()




# pass it a single corpus, rather than dict of all corpuses.
# prints tag info
def print_processed_corpus(corpus, num=5):
    for doc in corpus[:num]:
        for sent in doc.sentences:
            for word in sent.words: 
                print(f'word: {word.text}\t\tupos: {word.upos}\txpos: {word.xpos}')
            print("########################################")
        print('\n\n')
    

def print_processed_corpuses(all_corpuses, num=5):
    for lang in all_corpuses:
        lang_corpuses = all_corpuses[lang] # all corpuses of given language (src or trg)
        for corpus_name in lang_corpuses:
            corpus = lang_corpuses[corpus_name] # single corpus of given language
            print_processed_corpus(corpus, num)





# rather than re-running the tokenizer, segmenter, and pos_tagger on each corpus each time, just do so once, then store the corpuses (which now, rather than each being a list of str(sentence)'s, is now a list of Document objects). now, can  load them from pickle files
def store_corpuses(corpuses, filename):
    print('saving processed corpuses to pickle files...')
    dump(corpuses, open(filename, 'wb'))
    print('done.')

def load_corpuses(filename):
    print('loading processed corpuses...')
    return load(open(filename, 'rb'))






# each sentence of the corpus corresponds to a Document object in stanza.
# Document object consists of one or more Sentence objects. (in case a src or trg sentence of the corpus is actually more than one sentence)

def decase_all_corpuses(all_corpuses, num=5):
    decase_corpuses(all_corpuses["src"], num)    
    decase_corpuses(all_corpuses["trg"], num)


def decase_corpuses(corpuses, num=5):
    for corpus in corpuses:
        decase(corpuses[corpus], num)


# decase all words at cap_locations unless they are proper nouns, or are the pronoun 'I'.
def decase(corpus, num=5):
    #print(corpus[:num])
    for doc in corpus[:num]:
        for sent in doc.sentences:
            # print(sent.words)
            # # handle first word of each sent separately, which is always a cap_location
            # first_word = sent.words[0] # assumes each sent is non-
            # print(0, first_word.text)
            # if first_word.upos != 'PROPN' and first_word.text != 'I':
            #     sent.words[0].text = first_word.text.lower()
            # print(0, sent.words[0].text)
            # print()
            # remaining cap_locations include the sent position immediately after a double-quote or a colon (which I refer to as 'cap_location prefixes')
            cap_prefixes = ['"', ':']
            # need to start enumerating with i = 1, not i = 0
            #for i, word in enumerate(sent.words[1:]):
            # for i, word in enumerate(sent.words)[1:]:
            for i, word in enumerate(sent.words):
                if i == 0:
                    first_word = sent.words[0]
                    #print(i, first_word.text)
                    if first_word.upos != 'PROPN' and first_word.text != 'I':
                        sent.words[0].text = first_word.text.lower()
                    #print(i, sent.words[0].text)
                    #print()
                else:
                    #print(i, sent.words[i].text)
                    prev_word = sent.words[i-1]
                    if prev_word.text in cap_prefixes:
                        if word.upos != 'PROPN' and word.text != 'I':
                            sent.words[i].text = word.text.lower()
                    #print(i, sent.words[i].text)
                    #print()