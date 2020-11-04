import pytest
import torch

from src.import_configs import import_configs
from src.preprocessing.preprocess import construct_model_data, retrieve_model_data


def test_get_batches(
    checkpoint_path='/content/gdrive/My Drive/NMT/unittests/checkpoints/',
    corpus_path='/content/gdrive/My Drive/NMT/unittests/toy_corpuses/',
    config_path='/content/gdrive/My Drive/NMT/configs/',
):
    hyperparams = import_configs(config_path=config_path)
    hyperparams["vocab_type"] = "word"
    hyperparams["trim_type"] = "top_k"
    hyperparams["src_k"] = 10
    hyperparams["trg_k"] = 10
    hyperparams["train_bsz"] = 2
    hyperparams["dev_bsz"] = 2

    # train_batches, dev_batches, test_batches, vocabs, ref_corpuses, hyperparams = construct_model_data("train.de", "train.en", hyperparams=hyperparams,
    #                     corpus_path=corpus_path, overfit=True, write=False
    #                     )
    construct_model_data("train.de", "train.en", hyperparams=hyperparams,
                        corpus_path=corpus_path, checkpoint_path=checkpoint_path,
                        overfit=True
                        )
    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)

    device = hyperparams["device"]
    src_word_to_idx = model_data["src_word_to_idx"]
    trg_word_to_idx = model_data["trg_word_to_idx"]
    train_batches = model_data["train_batches"]
    dev_batches = model_data["dev_batches"]


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
    #assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([4, 3], device=device)))
    #assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([1, 0], device=device)))
    ### move to cpu #######
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([4, 3])))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([1, 0])))
    ########################

    assert decoder_inputs['in'].tolist() == train_decoder_inputs_in1
    #assert torch.all(torch.eq(decoder_inputs['lengths'], torch.tensor([5, 4], device=device)))
    #### move to cpu ######
    assert torch.all(torch.eq(decoder_inputs['lengths'], torch.tensor([5, 4])))
    ########################
    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False,  True]], [[False, False, False, False]]], device=device)))

    assert decoder_targets.tolist() == train_decoder_targets1


    # train batch 2
    encoder_inputs, decoder_inputs, decoder_targets = train_batches[1]
    assert encoder_inputs['in'].tolist() == train_encoder_inputs_in2
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([5])))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0])))

    assert decoder_inputs['in'].tolist() == train_decoder_inputs_in2
    assert torch.all(torch.eq(decoder_inputs['lengths'], torch.tensor([2])))
    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False, False, False]]], device=device)))

    assert decoder_targets.tolist() == train_decoder_targets2


    ### dev_batches:
    # dev batch 1
    encoder_inputs, decoder_inputs, corpus_indices = dev_batches[0]
    assert encoder_inputs['in'].tolist() == dev_encoder_inputs_in1
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([5, 4])))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0, 1])))

    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False, False, False]], [[False, False, False, False,  True]]], device='cuda:0')))
    assert decoder_inputs['max_src_len'] == 5

    assert torch.all(torch.eq(corpus_indices, torch.tensor([2, 0], device=device)))


    # dev batch 2
    encoder_inputs, decoder_inputs, corpus_indices = dev_batches[1]
    assert encoder_inputs['in'].tolist() == dev_encoder_inputs_in2
    assert torch.all(torch.eq(encoder_inputs['sorted_lengths'], torch.tensor([3])))
    assert torch.all(torch.eq(encoder_inputs['idxs_in_sorted'], torch.tensor([0])))

    assert torch.all(torch.eq(decoder_inputs['mask'], torch.tensor([[[False, False, False]]], device=device)))
    assert decoder_inputs['max_src_len'] == 3

    assert torch.all(torch.eq(corpus_indices, torch.tensor([1], device=device)))