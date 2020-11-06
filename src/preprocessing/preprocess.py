import torch
from pickle import dump

from src.preprocessing.corpus_utils import get_references, read_tokenized_corpuses
from src.preprocessing.build_word_vocabs import build_word_vocabs
from src.preprocessing.build_subword_vocabs import build_subword_vocabs
from src.preprocessing.apply_vocab import apply_vocab
from src.preprocessing.build_batches import get_batches
from src.model_utils import initialize_model, initialize_optimizer, store_checkpoint
# -converts all preprocessed corpuses into tensors that can be directly
# passed to a model, and saves them to pickle files.
# -returns corresponding hyperparameters that can be used to instantiate
# a compatible model.

# construct train, dev and test data to be used by model during training
# and inference, so that corpus preprocessing ensured to be compatible
# with model hyperparameters, e.g., vocab type, bsz, etc.
# gets saved to same folder the checkpoints are stored.
def construct_model_data(*corpus_names,
        hyperparams={},
        corpus_path='/content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/',
        checkpoint_path='/content/gdrive/My Drive/NMT/checkpoints/my_model/',
        src_vocab_file='vocab.de',
        trg_vocab_file='vocab.en',
        overfit=False):
    
    vocab_type = hyperparams["vocab_type"]
    # which variants of preprocessed corpuses to load depends on vocab type.
    # each entry of corpuses is a list of sentences, where sentence is list of words.
    corpuses = read_tokenized_corpuses(*corpus_names, path=corpus_path, prefix=vocab_type+"_")
        
    # build vocabs
    if vocab_type in ["word"]:
        vocabs = build_word_vocabs(corpuses, hyperparams)
    elif vocab_type in ["subword_ind", "subword_joint", "subword_pos"]:
        vocabs = build_subword_vocabs(corpus_path, vocab_type, hyperparams["vocab_threshold"], src_vocab_file, trg_vocab_file)
    
    # now that know the vocab sizes, can treat them as hyperparameters.
    hyperparams["src_vocab_size"] = len(vocabs["src_word_to_idx"])
    hyperparams["trg_vocab_size"] = len(vocabs["trg_word_to_idx"])
    print(f"src vocab size: {hyperparams['src_vocab_size']}")
    print(f"trg vocab size: {hyperparams['trg_vocab_size']}")

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
    
    # store initial checkpoint for a model training session,
    # holding all mutable training session data.
    model = initialize_model(hyperparams)
    optimizer = initialize_optimizer(model, hyperparams)
    epoch = 0 # how many epochs the model has trained for
    epoch_loss = 0 # loss of last completed epoch
    bleu = 0 # bleu score of model of current epoch on dev set 
    prev_bleu = 0 # bleu score of model of previous epoch on dev set 
    best_bleu = 0 # best bleu score of model on earlier epoch
    bad_epochs_count = 0 # when reaches early_stopping_threshold, training terminates
    store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu,
            best_bleu, bad_epochs_count, checkpoint_path, "most_recent_model")
    
    # store corresponding preprocessing of the corpuses, vocab info, and other
    # immutable data used during training.
    model_data = {"train_batches":train_batches,
        "dev_batches":dev_batches,
        "test_batches":test_batches, # for making test predictions after model is trained
        "references":get_references(overfit=overfit),
        "trg_word_to_idx":vocabs["trg_word_to_idx"], # for use in test_batches
        "idx_to_trg_word":vocabs["idx_to_trg_word"], # for making dev predictions during training, and test predictions during demo
        "src_word_to_idx":vocabs["src_word_to_idx"], # for making test predictions during demo
        "hyperparams":hyperparams} 

    dump(model_data, open(f"{checkpoint_path}model_data.pkl", 'wb'))

    # so can easily observe which sets of hyperparameters give
    # rise to which model training stats, dev set bleu stats, etc.
    with open(f"{checkpoint_path}model_train_stats.txt", 'w') as f:
        for hp in hyperparams:
            f.write(f"{hp}: {hyperparams[hp]}")
            f.write('\n')
        f.write('\n\n\n\n\n')