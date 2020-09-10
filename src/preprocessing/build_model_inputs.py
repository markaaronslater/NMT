from pickle import load, dump

from build_word_vocabs import build_word_vocabs
from build_subword_vocabs import build_subword_vocabs
from apply_vocab import apply_vocab
from corpus import get_tokenized_corpuses

###NOTE: if want to use subword vocab, must run jointBPE.sh prior to running this function
# output: list of batches of tensors that can be directly fed to the encoder-decoder network (writes them to pickle files)
# !!!use argparse
# vocab_params = {vocab_type:"word",
#                 trim_type:"top_k",
#                 src_k:30000,
#                 trg_k:30000,
#                 src_thres:2,
#                 trg_thres:2 # ??need store anything for subword vocabs??
#                 }

def build_model_inputs(*corpus_names, vocab_params, path='/content/gdrive/My Drive/iwslt16_en_de/', num=None):
    assert vocab_params["vocab_type"] == "word" or vocab_params["vocab_type"] == "subword"

    prefix = "decased_" if vocab_params["vocab_type"] == "word" else "bpe_"
    corpuses, ref_corpuses = get_tokenized_corpuses(*corpus_names, path, prefix, num)
    dump(ref_corpuses, open(f"{path}{prefix}_references", 'wb')) # save them, in case won't use immediately
        
    add_start_end_tokens(corpuses) # does not get included in references

    # build vocabs
    if vocab_params["vocab_type"] == "word":
        vocabs = build_word_vocabs(corpuses, vocab_params)
    elif vocab_params["vocab_type"] == "subword":
        # get_vocabs is much simpler than word_level, bc all the work was already done by jointBPE.sh. just needs to read it from file. i.e., don't need to pass corpuses as param
        vocabs = build_subword_vocabs(path)

    # convert each corpus of words to corpus of indices
    apply_vocab(corpuses, vocabs, vocab_params["vocab_type"])

    

    get_batches()


    # !!!save batches and references to pickle files so can skip this step next time

    # wrt vocabulary, model just needs idx_to_trg_word for producing translations, and needs lengths of vocabs so can build embedding table
    return batches, vocabs["idx_to_trg_word"]






# prepend and append trg sentences with start-of-sentence token, <sos>, and end-of-sentence token, <eos>, respectively
def add_start_end_tokens(corpuses):
    ###???is this list concat a constant time op??? is it creating a copy???
    corpuses["train.en"] = [['<sos>'] + sent + ['<eos>'] for sent in corpuses["train.en"]]

