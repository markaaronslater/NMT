import pytest
import torch

from src.import_configs import import_configs, constrain_configs
from src.preprocessing.preprocess import construct_model_data
from src.train import train
from src.predict import predict


checkpoint_path = '/content/gdrive/My Drive/NMT/unittests/checkpoints/'
config_path = '/content/gdrive/My Drive/NMT/configs/'
corpus_path = '/content/gdrive/My Drive/NMT/unittests/first_ten_sentences/'

# all tests use model default unittesting configuration (see overwrite_configs()
# in import_configs.py), except for those overridden inside test method.

# overfit to first 10 sentences of training set
def test_default_word_model():
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, overfit=True, write=False
                        )

    predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

def test_default_subword_model():
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    hyperparams["vocab_type"] = "subword_joint"
    train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, overfit=True, write=False
                        )

    predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# ensure still works on cpu.
# must change runtime type to cpu before performing this test
# def test_default_word_model_cpu():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["vocab_type"] = "word"
#     hyperparams["trim_type"] = "top_k"
#     hyperparams["device"] = "cpu"
#     train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# simplest possible model.
# - unidirectional encoder.
# - no attention mechanism.
def test_uni_no_attn():
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    hyperparams["attention_fn"] = "none"

    constrain_configs(hyperparams) # ensure passes constraint-check
    train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, overfit=True, write=False
                        )

    predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# two-layer vanilla network with layer_to_layer decoder_init_scheme
def test_layer_to_layer_uni_no_attn():
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    hyperparams["enc_num_layers"] = 2
    hyperparams["dec_num_layers"] = 2
    hyperparams["decoder_init_scheme"] = "layer_to_layer"
    hyperparams["attention_fn"] = "none"
    hyperparams["bidirectional"] = False
    constrain_configs(hyperparams) # ensure passes constraint-check
    train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, overfit=True, write=False
                        )

    predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# two-layer vanilla network with final_to_first decoder_init_scheme
def test_final_to_first_uni_no_attn():
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    hyperparams["enc_num_layers"] = 2
    hyperparams["dec_num_layers"] = 2
    hyperparams["decoder_init_scheme"] = "final_to_first"
    hyperparams["attention_fn"] = "none"
    hyperparams["bidirectional"] = False
    constrain_configs(hyperparams) # ensure passes constraint-check
    train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, overfit=True, write=False
                        )

    predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    



# associate some epoch number with saved model, so can verify stored correct model.
def test_early_stopping():
    # set random seed
    pass





def predict_train_data(hyperparams, train_batches, dev_batches, dev_references, idx_to_trg_word, checkpoint_path):
    # should achieve ~zero loss:
    model, loss = train(hyperparams, train_batches, dev_batches, dev_references, idx_to_trg_word, checkpoint_path, save=False)
    assert loss < .01

    # greedy search should be able to perfectly predict the training data:
    bleu, preds_time, post_time = predict(model, dev_batches, dev_references, idx_to_trg_word, checkpoint_path)
    assert bleu >= 100

    # beam search should be able to perfectly predict the training data:
    model.decoder.set_inference_alg("beam_search")
    bleu, preds_time, post_time = predict(model, dev_batches, dev_references, idx_to_trg_word, checkpoint_path)
    assert bleu >= 100