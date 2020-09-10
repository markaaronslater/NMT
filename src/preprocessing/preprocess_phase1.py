from pickle import load, dump
from math import ceil
import stanza

from corpus import read_corpuses, is_src_corpus

# use stanfordnlp processor to segment, tokenize, and produce morphological data about each corpus, like part-of-speech, case, number, person, gender, etc., saving outputs to pickle files

# expects each listed file to consist of one sentence per line, where line i of trg corpus is the translation of line i of src corpus.
# stores outputs to pickle files.
def preprocess_phase1(*corpus_names, path='/content/gdrive/My Drive/iwslt16_en_de/', piece_size=10000, src_lang="de", trg_lang="en"):

    stanza.download(src_lang)
    stanza.download(trg_lang)

    src_processor = stanza.Pipeline(lang=src_lang, processors='tokenize,mwt,pos')
    trg_processor = stanza.Pipeline(lang=trg_lang, processors='tokenize,mwt,pos')

    corpuses = read_corpuses(path, *corpus_names)

    apply_stanfordnlp_processors(corpuses, src_processor, trg_processor, path, piece_size)




# stores each processed corpus in pieces, with one piece per pickle file
def apply_stanfordnlp_processors(corpuses, src_processor, trg_processor, path='/content/gdrive/My Drive/iwslt16_en_de/', piece_size=10000):

    # first, store the number of pieces that will be associated with each corpus, so know how many pickle files to load from at beginning of preprocess_phase2
    num_corpus_pieces = {corpus_name:ceil(len(corpuses[corpus_name]) / piece_size) for corpus_name in corpuses}
    print(f"pieces: {num_corpus_pieces}\n")
    dump(num_corpus_pieces, open(path + 'num_corpus_pieces', 'wb'))

    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            apply_stanfordnlp_processor(corpus_name, corpuses[corpus_name], src_processor, path, piece_size)
        else:
            apply_stanfordnlp_processor(corpus_name, corpuses[corpus_name], trg_processor, path, piece_size)


# tags, tokenizes, segments corpuses.
# default piece_size is large enough that dev and test sets all fit inside single piece, but train sets will each get split into roughly 20 chunks, that will get stored separately as "checkpoints" in case colab disconnects during inference, etc., and then will concatenate them together when ready to use them.
# corpus is a list of Document objects
def apply_stanfordnlp_processor(corpus_name, corpus, processor, path='/content/gdrive/My Drive/iwslt16_en_de/', piece_size=10000):
    for j in range(0, len(corpus), piece_size):
        corpus_piece = corpus[j:j+piece_size] # list of piece_size Document objects
        piece_number = j // piece_size + 1
        print(f"processing piece {piece_number} of {corpus_name}...")
        docs = [] # List[Document]
        for sentence in corpus_piece:
            docs.append(processor(sentence))

        dump(docs, open(f"{path}phase1_{corpus_name}_{piece_number}", 'wb'))


