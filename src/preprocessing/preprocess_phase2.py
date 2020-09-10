from pickle import load

from corpus import read_corpuses, write_corpuses, get_references, tokenize_corpuses
from decase import decase_corpuses

def preprocess_phase2(path='/content/gdrive/My Drive/iwslt16_en_de/', num=None):
    corpuses = get_preprocess_phase1_outputs() # corpuses is List[Document]
    decase_corpuses(corpuses, path, num) # now, corpuses is List[List[str]]
    write_corpuses(corpuses, path, 'decased_', num) # write to text files


def get_preprocess_phase1_outputs(*corpuses, path='/content/gdrive/My Drive/iwslt16_en_de/'):
    # get the number of pieces that each corpus is distributed across
    num_corpus_pieces = load(open(path + 'num_corpus_pieces', 'rb'))
    print(f"pieces: {num_corpus_pieces}\n")
    processed_corpuses = {}
    for corpus_name in corpuses:
        processed_corpuses[corpus_name] = merge_corpus_pieces(corpus_name, num_corpus_pieces[corpus_name], path)

    return processed_corpuses


# for each sentence of each corpus, nlpprocessor produced a Document object. for each corpus, only ~10000 sentences were processed, yielding List[Document]. This function concatenates all of these lists into a single List[Document] corresponding to the entire corpus.
def merge_corpus_pieces(corpus_name, num_pieces, path='/content/gdrive/My Drive/iwslt16_en_de/'):
    merged_corpus = []
    for i in range(1, num_pieces+1):
        #merged_corpus += load_corpus(f"{path}processed_{corpus_name}_{i}")
        merged_corpus += load(open(f"{path}processed_{corpus_name}_{i}", 'rb'))

    return merged_corpus
