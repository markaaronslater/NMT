from pickle import load, dump
from math import ceil

import stanza

from src.preprocessing.corpus_utils import is_src_corpus, read_corpuses, print_processed_corpus

# use stanfordnlp processor to perform
# -tokenization,
# -multi-word token expansion, and
# -morphological annotation (e.g., part-of-speech, case, number, person, gender)
# to each sentence of each corpus.
# this step is time and memory expensive, so saves outputs to pickle files,
# and processes corpuses in pieces that are later merged together.

# use their tokenizer without its accompanying sentence segmenter, bc
# would destroy alignment of source and target sentences,
# bc each line can actually consist of multiple sentences.
# (we could use a bsz of 1 to avoid that, but takes very long time).
def apply_stanza_processors(*corpus_names,
                            path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/',
                            src_lang="de", trg_lang="en", _start=1, num=None,
                            doc_size=10000, tok_bsz=64, mwt_bsz=200, pos_bsz=10000):
    stanza.download(lang=src_lang, processors='tokenize,mwt,pos')
    stanza.download(lang=trg_lang, processors='tokenize,pos')

    src_processor = stanza.Pipeline(lang=src_lang, processors='tokenize,mwt,pos', tokenize_no_ssplit=True, tokenize_batch_size=tok_bsz, mwt_batch_size=mwt_bsz, pos_batch_size=pos_bsz)
    trg_processor = stanza.Pipeline(lang=trg_lang, processors='tokenize,pos', tokenize_no_ssplit=True, tokenize_batch_size=tok_bsz, pos_batch_size=pos_bsz)

    corpuses = read_corpuses(*corpus_names, path=path, _start=_start, num=num)

    # store the number of pieces that each corpus's stanza processor outputs
    # will be stored across (so know how many pickle files to merge together
    # in downstream processing)
    num_corpus_pieces = {corpus_name:ceil(len(corpuses[corpus_name]) / doc_size) for corpus_name in corpuses}
    print(f"pieces: {num_corpus_pieces}\n")
    stanza_path = path + 'stanza_outputs/'
    dump(num_corpus_pieces, open(f"{stanza_path}num_corpus_pieces.pkl", 'wb'))

    for corpus_name in corpuses:
        if is_src_corpus(corpus_name):
            apply_stanza_processor(corpus_name, corpuses[corpus_name], src_processor, path=stanza_path, doc_size=doc_size)
        else:
            apply_stanza_processor(corpus_name, corpuses[corpus_name], trg_processor, path=stanza_path, doc_size=doc_size)
    print("done.")


# creates massive list that does not fit in RAM, so process the corpus 
# <doc_size> sentences at a time (e.g., if doc_size=10,000, then English
# train set, which is ~200,000, gets processed as ~20 pieces). each piece saved
# to separate pickle file, so when retrieve outputs, must merge them back
# into single processed corpus.
def apply_stanza_processor(corpus_name, corpus, processor, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/stanza_outputs/', doc_size=10000):
    for j in range(0, len(corpus), doc_size):
        # supply input in format required for disabling sentence segmentation in tokenizer
        corpus_piece = '\n\n'.join(corpus[j:j+doc_size])
        piece_number = j // doc_size + 1
        print(f"processing piece {piece_number} of {corpus_name}...")
        doc = processor(corpus_piece) # returns Document object
        #print_processed_corpus(doc.sentences) 
        # only need sentences (list of Stanza Sentence objects).
        dump(doc.sentences, open(f"{path}stanza_{corpus_name}_{piece_number}.pkl", 'wb'))


# this might not fit in memory. in that case, do as truecase_corpuses() does,
# and load and process single piece at a time.
# def merge_corpus_pieces(corpus_name, num_pieces, path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/stanza_outputs/'):
#     merged_corpus = []
#     for piece_number in range(1, num_pieces+1):
#         merged_corpus += load(open(f"{path}stanza_{corpus_name}_{piece_number}.pkl", 'rb'))

#     return merged_corpus