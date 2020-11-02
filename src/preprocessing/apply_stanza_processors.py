from pickle import load, dump
import stanza

from src.preprocessing.corpus_utils import is_src_corpus

# use stanfordnlp processor to perform
# -tokenization,
# -multi-word token expansion, and
# -morphological annotation (e.g., part-of-speech, case, number, person, gender)
# to each sentence of each corpus.
# this step is expensive, so save outputs to pickle files.

# use their tokenizer without its accompanying sentence segmenter, bc
# would destroy alignment of source and target sentences,
# bc each line can actually consist of multiple sentences.
# (we could use a bsz of 1 to avoid that, but takes very long time).
def apply_stanza_processors(*corpus_names,
                            path='/content/gdrive/My Drive/NMT/iwslt16_en_de/',
                            src_lang="de", trg_lang="en", _start=1, num=None):
    stanza.download(lang=src_lang, processors='tokenize,mwt,pos')
    stanza.download(lang=trg_lang, processors='tokenize,pos')

    src_processor = stanza.Pipeline(lang=src_lang, processors='tokenize,mwt,pos', tokenize_no_ssplit=True)
    trg_processor = stanza.Pipeline(lang=trg_lang, processors='tokenize,pos', tokenize_no_ssplit=True)

    for corpus_name in corpus_names:
        with open(path + corpus_name, mode='rt', encoding='utf-8') as f:
            corpus = f.read().strip().split('\n')
            upper = num if num is not None else len(corpus)
            start = _start-1 # convert to 0-based idxing    
            # supply input in format required for disabling sentence segmentation.
            processor_input = '\n\n'.join(corpus[start:start+upper])

            if is_src_corpus(corpus_name):
                # pass entire corpus. stanza will process it in batches.
                doc = src_processor(processor_input) # returns Document object
                dump(doc, open(f"{path}stanza_{corpus_name}.pkl", 'wb'))
            else:
                doc = trg_processor(processor_input)
                dump(doc, open(f"{path}stanza_{corpus_name}.pkl", 'wb'))


# returns a dict of corpus_names mapped to stanza Document objects
def retrieve_stanza_outputs(*corpus_names,
                            path='/content/gdrive/My Drive/NMT/iwslt16_en_de/'):
    corpuses = {}
    for corpus_name in corpus_names:
        corpuses[corpus_name] = load(open(f"{path}stanza_{corpus_name}.pkl", 'rb'))

    return corpuses