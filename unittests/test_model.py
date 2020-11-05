import pytest
import torch

from src.import_configs import import_configs, constrain_configs
from src.preprocessing.corpus_utils import get_references
from src.preprocessing.preprocess import construct_model_data, retrieve_model_data
from src.train import train
from src.predict import predict
from src.evaluate import evaluate


# all tests use model default unittesting configuration (see overwrite_configs()
# in import_configs.py), except for those overridden inside test method.

# overfit to first 10 sentences of training set
# overview: scaled_dot_product attn, 1 lstm layer, attention layer, tied weights.
def test_default_word_model(checkpoint_path='/content/gdrive/My Drive/NMT/unittests/checkpoints/',
                config_path='/content/gdrive/My Drive/NMT/configs/',
                corpus_path = '/content/gdrive/My Drive/NMT/unittests/first_ten_sentences/'
):
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    # use word-level vocab
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, checkpoint_path=checkpoint_path, 
                        overfit=True
                        )

    # model of sufficient capacity should be able to bring loss down to ~zero.
    model, loss = train(total_epochs=100, early_stopping=False, checkpoint_path=checkpoint_path, save=False, write=False)
    assert loss < .01

    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    dev_batches = model_data["dev_batches"] # holds the training data, bc overfit=True
    dev_references = model_data["references"] # holds the training data, bc overfit=True
    idx_to_trg_word = model_data["idx_to_trg_word"]

    # greedy search should be able to perfectly predict the training data.
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100

    # beam search should be able to perfectly predict the training data.
    model.decoder.set_inference_alg("beam_search")
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100


def test_default_subword_model(checkpoint_path='/content/gdrive/My Drive/NMT/unittests/checkpoints/',
                config_path='/content/gdrive/My Drive/NMT/configs/',
                corpus_path = '/content/gdrive/My Drive/NMT/unittests/first_ten_sentences/'
):
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    # use subword-level vocab
    hyperparams["vocab_type"] = "subword_joint"
    hyperparams["learning_rate"] = .01 # increase learning rate
    print(f"learning_rate: {hyperparams['learning_rate']}")

    construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, checkpoint_path=checkpoint_path, 
                        overfit=True
                        )

    # model of sufficient capacity should be able to bring loss down to ~zero.
    model, loss = train(total_epochs=100, early_stopping=False, checkpoint_path=checkpoint_path, save=False, write=True)
    assert loss < .01

    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    dev_batches = model_data["dev_batches"] # holds the training data, bc overfit=True
    dev_references = model_data["references"] # holds the training data, bc overfit=True
    idx_to_trg_word = model_data["idx_to_trg_word"]

    # greedy search should be able to perfectly predict the training data.
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100

    # beam search should be able to perfectly predict the training data.
    model.decoder.set_inference_alg("beam_search")
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100


def test_outdrop(checkpoint_path='/content/gdrive/My Drive/NMT/unittests/checkpoints/',
                config_path='/content/gdrive/My Drive/NMT/configs/',
                corpus_path = '/content/gdrive/My Drive/NMT/unittests/first_ten_sentences/'
):
    hyperparams = import_configs(config_path=config_path, unittesting=True)
    # use word-level vocab
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    hyperparams["enc_dropout"] = .5
    hyperparams["dec_dropout"] = .5
    print(f"hidden size: {hyperparams['dec_hidden_size']}")

    construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, checkpoint_path=checkpoint_path, 
                        overfit=True
                        )

    # model of sufficient capacity should be able to bring loss down to ~zero.
    model, loss = train(total_epochs=100, early_stopping=False, checkpoint_path=checkpoint_path, save=False, write=False)
    assert loss < .01

    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    dev_batches = model_data["dev_batches"] # holds the training data, bc overfit=True
    dev_references = model_data["references"] # holds the training data, bc overfit=True
    idx_to_trg_word = model_data["idx_to_trg_word"]

    # greedy search should be able to perfectly predict the training data.
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100

    # beam search should be able to perfectly predict the training data.
    model.decoder.set_inference_alg("beam_search")
    dev_translations, _, _ = predict(model, dev_batches, idx_to_trg_word, checkpoint_path)
    bleu = evaluate(dev_translations, dev_references)
    assert bleu >= 100


# def test_default_subword_model():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["vocab_type"] = "subword_joint"
#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # default word model, except dn divide scores by scaling factor inside attention fn.
# def test_attn():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["vocab_type"] = "word"
#     hyperparams["trim_type"] = "top_k"
#     hyperparams["attention_fn"] = "dot_product"
#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # no weight tying, no additional attention layer
# def test_no_tying():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["vocab_type"] = "word"
#     hyperparams["trim_type"] = "top_k"
#     hyperparams["attention_layer"] = False
#     hyperparams["tie_weights"] = False

#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # no weight tying and no attention mechanism.
# def test_no_attn_no_tying():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["vocab_type"] = "word"
#     hyperparams["trim_type"] = "top_k"
#     hyperparams["attention_fn"] = "none"
#     hyperparams["attention_layer"] = False
#     hyperparams["tie_weights"] = False

#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # default model, except dropout after lstm is turned on.
# def test_dropout():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["enc_dropout"] = 0.2
#     hyperparams["dec_dropout"] = 0.2

#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # ensure still works on cpu.
# # must change runtime type to cpu before performing this test
# # def test_default_word_model_cpu():
# #     hyperparams = import_configs(config_path=config_path, unittesting=True)
# #     hyperparams["vocab_type"] = "word"
# #     hyperparams["trim_type"] = "top_k"
# #     hyperparams["device"] = "cpu"
# #     train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
# #                         corpus_path=corpus_path, overfit=True, write=False
# #                         )

# #     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # simplest possible model.
# # - unidirectional encoder.
# # - no attention mechanism.
# def test_uni_no_attn():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["attention_fn"] = "none"

#     constrain_configs(hyperparams) # ensure passes constraint-check
#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # two-layer vanilla network with layer_to_layer decoder_init_scheme
# def test_layer_to_layer_uni_no_attn():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["enc_num_layers"] = 2
#     hyperparams["dec_num_layers"] = 2
#     hyperparams["decoder_init_scheme"] = "layer_to_layer"
#     hyperparams["attention_fn"] = "none"
#     hyperparams["bidirectional"] = False
#     constrain_configs(hyperparams) # ensure passes constraint-check
#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    

# # two-layer vanilla network with final_to_first decoder_init_scheme
# def test_final_to_first_uni_no_attn():
#     hyperparams = import_configs(config_path=config_path, unittesting=True)
#     hyperparams["enc_num_layers"] = 2
#     hyperparams["dec_num_layers"] = 2
#     hyperparams["decoder_init_scheme"] = "final_to_first"
#     hyperparams["attention_fn"] = "none"
#     hyperparams["bidirectional"] = False
#     constrain_configs(hyperparams) # ensure passes constraint-check
#     train_batches, dev_batches, vocabs, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
#                         corpus_path=corpus_path, overfit=True, write=False
#                         )

#     predict_train_data(hyperparams, train_batches, dev_batches, ref_corpuses["train.en"], vocabs["idx_to_trg_word"], checkpoint_path)
    



# # associate some epoch number with saved model, so can verify stored correct model.
# def test_early_stopping():
#     # set random seed
#     pass