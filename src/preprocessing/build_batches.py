import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# convert all train/dev data into padded batches of tensors.
def get_batches(corpuses, train_bsz=64, dev_bsz=32, device="cuda:0", overfit=False):
    # train_batches is a list of (encoder_inputs, decoder_inputs, decoder_targets) triples.
    start_time = time.time()
    train_batches = get_train_batches(corpuses["train.de"], corpuses["train.en"], train_bsz, device)
    print(f"took {time.time()-start_time} seconds to produce train batches")

    # dev_batches and test_batches each use same type of batch.
    # dev_batches is a list of (encoder_inputs, decoder_inputs, corpus_indices) triples.
    start_time = time.time()
    # when unit testing, dev set consists of train sentences, bc goal
    # is to overfit such that perfectly predicts sentences it trained on.
    dev_batches = get_test_batches(corpuses["dev.de"], dev_bsz, device) if not overfit else get_test_batches(corpuses["train.de"], dev_bsz, device)
    print(f"took {time.time()-start_time} seconds to produce dev batches")

    return train_batches, dev_batches


def get_train_batches(src_sentences, trg_sentences, bsz, device):
    # sort each src_sentence, trg_sentence pair ahead of time
    # (arbitrarily chose to sort by trg length) for intelligent batching
    # (most sequences within a batch are same length,
    # minimizing number of pad tokens).
    training_pairs = sorted(list(zip(src_sentences, trg_sentences)), key = lambda pair: len(pair[1]), reverse=True)
    train_batches = [get_train_batch(training_pairs[i:i+bsz], device) for i in range(0, len(training_pairs), bsz)]

    return train_batches


# <batch> is a list of <bsz> (source sentence, target sentence) pairs,
# each of which is represented as a list of indices in the vocabs.
def get_train_batch(batch, device):
    bsz = len(batch) # last batch could have diff size, if dataset length not divisible by batch size
    # "unzip" the pairs back into separate lists.
    src_batch, trg_batch = [pair[0] for pair in batch], [pair[1] for pair in batch]

    encoder_inputs = src_batch
    decoder_inputs = [trg_batch[i][:-1] for i in range(bsz)] # dn include <eos>
    decoder_targets = [trg_batch[i][1:] for i in range(bsz)] # dn include <sos>
    # "targets", in this case, refers to ground truth tokens of target sentence.

    src_lengths = torch.tensor([len(src_sent) for src_sent in encoder_inputs]).long()
    trg_lengths = torch.tensor([len(trg_sent) for trg_sent in decoder_targets]).long()
    
    # determine the length that each sentence of a batch must be padded to.
    max_src_len = src_lengths.max()
    max_trg_len = trg_lengths.max()

    padded_encoder_inputs = torch.zeros(bsz, max_src_len, device=device).long() 
    padded_decoder_inputs = torch.zeros(bsz, max_trg_len, device=device).long() 
    padded_decoder_targets = torch.zeros(bsz, max_trg_len, device=device).long()

    for i in range(bsz):
        padded_encoder_inputs[i,:src_lengths[i]] = torch.tensor(encoder_inputs[i]).long()
        padded_decoder_inputs[i,:trg_lengths[i]] = torch.tensor(decoder_inputs[i]).long()
        padded_decoder_targets[i,:trg_lengths[i]] = torch.tensor(decoder_targets[i]).long()

    # (in DESCENDING order).
    sorted_src_lengths, idxs_in_encoder_inputs = src_lengths.sort(0, descending=True)
    # idxs_in_encoder_inputs holds, at position i, the index of the
    # i'th longest source sentence within padded_encoder_inputs.
    # -> use it to sort padded_encoder_inputs in descending order:
    padded_encoder_inputs = padded_encoder_inputs[idxs_in_encoder_inputs]
    
    _, idxs_in_sorted_encoder_inputs = idxs_in_encoder_inputs.sort(0)
    # idxs_in_sorted_encoder_inputs holds, at position i, the index within
    # the now sorted padded_encoder_inputs, of the i'th sentence in the
    # original (unsorted) padded_encoder_inputs.
    # (in ASCENDING order).
    # -> use it to "unsort" the resultant encodings after the encoder fwd pass,
    # so that they correspond to the sorted target sentences in decoder_inputs
    # during decoder fwd pass.

    # construct mask to be used by attention mechanism of decoder.
    mask = torch.ones(bsz, max_src_len, device=device) == 1
    # build using unsorted src_lengths, so that lines up with corresponding
    # target sentences inside decoder.
    for i in range(bsz):
        mask[i, :src_lengths[i]] = 0
    mask = mask.view(bsz, 1, max_src_len)

    # -pass sorted lengths, so can pack sequences when pass thru lstm.
    # -pass idxs in sorted encoder inputs, so can unsort the sequences after
    # pass thru lstm, so that they line up with target sentences.
    encoder_inputs = {
        "in":padded_encoder_inputs,
        "sorted_lengths":sorted_src_lengths,
        "idxs_in_sorted":idxs_in_sorted_encoder_inputs
    }

    decoder_inputs = {
        "in":padded_decoder_inputs,
        "lengths":trg_lengths,
        "mask":mask}

    # can pack targets up ahead of time (will compute loss wrt packed preds).
    decoder_targets = pack_padded_sequence(padded_decoder_targets, trg_lengths, batch_first=True)
    decoder_targets = decoder_targets.data

    return encoder_inputs, decoder_inputs, decoder_targets


# use this same function for preparing dev and test batches.
def get_test_batches(src_sentences, bsz, device):
    test_triples = [(idx, len(sent), sent) for idx, sent in enumerate(src_sentences)]
    # -sort by src_length here bc there are no targets during inference
    # (maybe was confusing that I sorted training pairs by TARGET length).
    # -this allows us to intelligently batch so that can minimize number of
    # pad tokens when decoding a batch of src sentences during inference,
    # and maximize likelihood that each predicted translation of batch
    # requires similar number of decoding steps.
    # -keeps track of orig index in corpus so can later "unsort".
    test_triples = sorted(test_triples, key = lambda triple: triple[1], reverse=True)
    test_batches = [get_test_batch(test_triples[i:i+bsz], device) for i in range(0, len(src_sentences), bsz)]

    return test_batches


def get_test_batch(batch, device):
    bsz = len(batch) 
    src_lengths = torch.tensor([triple[1] for triple in batch]).long()

    # so that can unsort predicted translations.
    corpus_indices = torch.tensor([triple[0] for triple in batch], device=device).long()

    max_src_len = src_lengths.max()
    padded_encoder_inputs = torch.zeros(bsz, max_src_len, device=device).long() 
    for i in range(bsz):
        padded_encoder_inputs[i,:src_lengths[i]] = torch.tensor(batch[i][2]).long()
        
    # perform so that meet encoder's spec.
    # !!!change so that this is no longer necessary.
    sorted_src_lengths, idxs_in_encoder_inputs = src_lengths.sort(0, descending=True)
    padded_encoder_inputs = padded_encoder_inputs[idxs_in_encoder_inputs] 
    src_lengths = sorted_src_lengths
    _, idxs_in_sorted_encoder_inputs = idxs_in_encoder_inputs.sort(0)

    # construct mask to be used by attention mechanism of decoder.
    # initialize all entries with 1, and then overwrite non-pad entries with 0.
    mask = torch.ones(bsz, max_src_len, device=device) == 1
    for i in range(bsz):
        mask[i, :src_lengths[i]] = 0
    mask = mask.view(bsz, 1, max_src_len) 

    encoder_inputs = {
        "in":padded_encoder_inputs,
        "sorted_lengths":sorted_src_lengths,
        "idxs_in_sorted":idxs_in_sorted_encoder_inputs
    }

    # max_src_len is used by decoder, along with decode_slack,
    # to heuristically decide when to stop decoding.
    decoder_inputs = {
        "mask":mask,
        "max_src_len": max_src_len
    }

    return encoder_inputs, decoder_inputs, corpus_indices