import torch
from pickle import load, dump

from src.preprocessing.corpus_utils import get_references, read_tokenized_corpuses
from src.preprocessing.build_word_vocabs import build_word_vocabs
from src.preprocessing.build_subword_vocabs import build_subword_vocabs
from src.preprocessing.apply_vocab import apply_vocab
from src.preprocessing.build_batches import get_batches

# -converts all preprocessed corpuses into tensors that can be directly
# passed to a model, and saves them to pickle files.
# -returns corresponding hyperparameters that can be used to instantiate
# a compatible model.
def construct_model_data(*corpus_names,
        hyperparams={},
        corpus_path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/',
        data_path='/content/gdrive/My Drive/NMT/data/',
        model_name='my_model',
        src_vocab_file='vocab.de',
        trg_vocab_file='vocab.en',
        overfit=False,
        write=True):
    
    vocab_type = hyperparams["vocab_type"]
    # which variants of preprocessed corpuses to load depends on vocab type.
    corpuses = read_tokenized_corpuses(*corpus_names, path=corpus_path, prefix=vocab_type+"_")
        
    # build vocabs
    if vocab_type in ["word"]:
        vocabs = build_word_vocabs(corpuses, hyperparams)
    elif vocab_type in ["subword_ind", "subword_joint", "subword_pos"]:
        vocabs = build_subword_vocabs(corpus_path, vocab_type, hyperparams["vocab_threshold"], src_vocab_file, trg_vocab_file)
    
    # now that know the vocab sizes, can treat them as hyperparameters.
    hyperparams["src_vocab_size"] = len(vocabs["src_word_to_idx"])
    hyperparams["trg_vocab_size"] = len(vocabs["trg_word_to_idx"])
    # not technically hyperparams, but include special indices for convenience:
    sos_idx = vocabs["trg_word_to_idx"]["<sos>"]
    eos_idx = vocabs["trg_word_to_idx"]["<eos>"]
    hyperparams["sos_idx"] = sos_idx
    hyperparams["eos_idx"] = eos_idx

    # convert each corpus of words to corpus of indices, and replace
    # out-of-vocabulary words with unknown token (if using word-level vocabs).
    apply_vocab(corpuses, vocabs, vocab_type)
    
    # only target sentences use start and end-of-sentence tokens
    corpuses["train.en"] = [[sos_idx] + sent + [eos_idx] for sent in corpuses["train.en"]]
    
    # package corpuses up into batches of model inputs, along with other necessary
    # data, such as masks for attention mechanism, lengths for efficient
    # packing/unpacking of PackedSequence objects, etc.
    train_batches, dev_batches, test_batches = get_batches(corpuses, train_bsz=hyperparams["train_bsz"], dev_bsz=hyperparams["dev_bsz"], test_bsz=hyperparams["test_bsz"], device=hyperparams["device"], overfit=overfit)
    
    # lastly, even though independent of model, include the references for convenience.
    ref_corpuses = get_references(overfit=overfit)

    # can directly be loaded to instantiate and then train a model.
    model_data = {
        "train_batches":train_batches,
        "dev_batches":dev_batches,
        "test_batches":test_batches,
        "idx_to_trg_word":vocabs["idx_to_trg_word"],
        "ref_corpuses":ref_corpuses,
        "hyperparams":hyperparams
    }

    if write:
        dump(model_data, open(f"{data_path}{model_name}.pkl", 'wb'))

    # for convenience in unit tests
    return train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams


def retrieve_model_data(data_path='/content/gdrive/My Drive/NMT/data/', model_name='my_model'):
    return load(open(f"{data_path}{model_name}.pkl", 'rb'))