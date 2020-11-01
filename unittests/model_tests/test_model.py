import pytest # so can use pytest.raises() method
import torch

from NMT.src.import_configs import import_configs
from NMT.src.preprocessing.preprocess import construct_model_data, retrieve_model_data
from NMT.src.train import train
from NMT.src.predict import predict
from NMT.src.preprocessing.corpus_utils import read_tokenized_corpuses





def test_get_batches():
    path = '/content/gdrive/My Drive/NMT/'
    corpus_path = path + 'corpuses/toy_corpuses/'
    config_path = path + 'configs/'
    data_path = path + 'data/'
    checkpoint_path = path + 'checkpoints/'
    model_name = 'my_model' 

    hyperparams = import_configs(config_path=config_path)
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    hyperparams["src_k"] = 10
    hyperparams["trg_k"] = 10
    hyperparams["train_bsz"] = 2
    hyperparams["dev_bsz"] = 2
    hyperparams["decode_slack"] = 30
    hyperparams["early_stopping"] = False

    vocabs, corpuses, ref_corpuses = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, data_path=data_path, model_name=model_name, overfit=True
                        )

    model_data = retrieve_model_data(data_path=data_path, model_name=model_name)
    train_batches = model_data["train_batches"]
    dev_batches = model_data["dev_batches"]
    test_batches = model_data["test_batches"]
    idx_to_trg_word = model_data["idx_to_trg_word"]
    ref_corpuses = model_data["ref_corpuses"]
    hyperparams = model_data["hyperparams"]
    device = hyperparams["device"]

    #print(f'src vocab:{vocabs["src_word_to_idx"]}')
    #print(f'trg vocab:{vocabs["trg_word_to_idx"]}')
    src_word_to_idx = vocabs["src_word_to_idx"]
    trg_word_to_idx = vocabs["trg_word_to_idx"]

    train_encoder_inputs_in1 = [['das', 'ist', 'wahr', '.'], ['mache', 'ich', 'Ja', '<pad>']]
    train_encoder_inputs_in1 = [[src_word_to_idx[word] for word in sent] for sent in train_encoder_inputs_in1]
    train_encoder_inputs_in2 = [['heute', 'Abend', '!', '!', '!']]
    train_encoder_inputs_in2 = [[src_word_to_idx[word] for word in sent] for sent in train_encoder_inputs_in2]

    train_decoder_inputs_in1 = [['<sos>', 'do', 'I', '?', 'yes'], ['<sos>', 'it', "'s", 'true', '<pad>']]
    train_decoder_inputs_in1 = [[trg_word_to_idx[word] for word in sent] for sent in train_decoder_inputs_in1]
    train_decoder_inputs_in2 = [['<sos>', 'tonight']]
    train_decoder_inputs_in2 = [[trg_word_to_idx[word] for word in sent] for sent in train_decoder_inputs_in2]

    train_decoder_targets1 = ['do', 'it', 'I', "'s", '?', 'true', 'yes', '<eos>', '<eos>']
    train_decoder_targets1 = [trg_word_to_idx[word] for word in train_decoder_targets1]
    train_decoder_targets2 = ['tonight', '<eos>']
    train_decoder_targets2 = [trg_word_to_idx[word] for word in train_decoder_targets2]

    dev_encoder_inputs_in1 = [['heute', 'Abend', '!', '!', '!'], ['das', 'ist', 'wahr', '.', '<pad>']]
    dev_encoder_inputs_in1 = [[src_word_to_idx[word] for word in sent] for sent in dev_encoder_inputs_in1]
    dev_encoder_inputs_in2 = [['mache', 'ich', 'Ja']]
    dev_encoder_inputs_in2 = [[src_word_to_idx[word] for word in sent] for sent in dev_encoder_inputs_in2]


    ### train_batches:
    # train batch 1
    encoder_inputs, decoder_inputs, decoder_targets = train_batches[0]
    assert encoder_inputs['in'].tolist() == train_encoder_inputs_in1
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([4, 3], device=device)))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([1, 0], device=device)))

    assert decoder_inputs['in'].tolist() == train_decoder_inputs_in1
    assert torch.all(torch.eq(decoder_inputs['lengths'], torch.tensor([5, 4], device=device)))
    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False,  True]], [[False, False, False, False]]], device=device)))

    assert decoder_targets.tolist() == train_decoder_targets1


    # train batch 2
    encoder_inputs, decoder_inputs, decoder_targets = train_batches[1]
    assert encoder_inputs['in'].tolist() == train_encoder_inputs_in2
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([5], device=device)))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0], device=device)))

    assert decoder_inputs['in'].tolist() == train_decoder_inputs_in2
    assert torch.all(torch.eq(decoder_inputs['lengths'], torch.tensor([2], device=device)))
    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False, False, False]]], device=device)))

    assert decoder_targets.tolist() == train_decoder_targets2


    ### dev_batches:
    # dev batch 1
    encoder_inputs, decoder_inputs, corpus_indices = dev_batches[0]
    assert encoder_inputs['in'].tolist() == dev_encoder_inputs_in1
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([5, 4], device=device)))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0, 1], device=device)))

    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False, False, False]], [[False, False, False, False,  True]]], device='cuda:0')))
    assert decoder_inputs['max_src_len'] == 5

    assert torch.all(torch.eq(corpus_indices, torch.tensor([2, 0], device=device)))


    # dev batch 2
    encoder_inputs, decoder_inputs, corpus_indices = dev_batches[1]
    assert encoder_inputs['in'].tolist() == dev_encoder_inputs_in2
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([3], device=device)))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0], device=device)))

    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False]]], device=device)))
    assert decoder_inputs['max_src_len'] == 3

    assert torch.all(torch.eq(corpus_indices, torch.tensor([1], device=device)))















# overfit to first 10 sentences of training set
def test_word_model():
    # recommended: place cloned NMT folder in Google drive folder 'My Drive':
    path = '/content/gdrive/My Drive/NMT/'
    corpus_path = path + 'corpuses/iwslt16_en_de/'
    config_path = path + 'configs/'
    data_path = path + 'data/'
    checkpoint_path = path + 'checkpoints/'
    model_name = 'my_model' # name of model tensor batches, hyperparameters, etc., saved as pickle file inside data_path

    hyperparams = import_configs(config_path=config_path)
    # overwrite hyperparams to conform to test conditions
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    hyperparams["src_k"] = 200 # set large enough such that no <unk> tokens (or else will not achieve BLEU of 100)
    hyperparams["trg_k"] = 200
    hyperparams["train_bsz"] = 3
    hyperparams["dev_bsz"] = 3
    hyperparams["decode_slack"] = 30 # set large enough such that can finish predicting each of the 10 target sentences (or else will not achieve BLEU of 100)
    hyperparams["early_stopping"] = False # let the loss go down to zero.
    hyperparams["total_epochs"] = 50
    hyperparams["enc_hidden_size"] = 1000 # ensure model is of sufficient capacity
    hyperparams["dec_hidden_size"] = 1000
    hyperparams["enc_dropout"] = 0 # ensure regularization turned off
    hyperparams["dec_dropout"] = 0
    hyperparams["L2_reg"] = 0

    vocabs, corpuses, ref_corpuses = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, data_path=data_path, model_name=model_name, overfit=True
                        )
    model_data = retrieve_model_data(data_path=data_path, model_name=model_name)
    train_batches = model_data["train_batches"]
    dev_batches = model_data["dev_batches"]
    idx_to_trg_word = model_data["idx_to_trg_word"]
    ref_corpuses = model_data["ref_corpuses"]
    hyperparams = model_data["hyperparams"]

    #print(f'src vocab:{vocabs["src_word_to_idx"]}')
    #print(f'trg vocab:{vocabs["trg_word_to_idx"]}')
    dev_references = ref_corpuses["train.en"] # predict the training data

    # should achieve ~zero loss:
    model, loss = train(hyperparams, train_batches, dev_batches, dev_references, idx_to_trg_word, checkpoint_path, save=False)
    assert loss < .01

    # greedy search should be able to perfectly predict the training data:
    bleu, preds_time, post_time = predict(model, dev_batches, dev_references, idx_to_trg_word, checkpoint_path)
    assert bleu == 100

    # beam search should be able to perfectly predict the training data:
    model.decoder.set_inference_alg("beam_search")
    bleu, preds_time, post_time = predict(model, dev_batches, dev_references, idx_to_trg_word, checkpoint_path)
    assert bleu == 100


def test_subword_model():





def test_early_stopping():
    # set seed so know when done.
    pass