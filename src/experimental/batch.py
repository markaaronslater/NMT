import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# constructs all the data structures used in a single minibatch training step.
# batch_srcSentIndices is a list of <bsz> src sentences, each represented as a list of word indices
# batch_trgSentIndices is a list of <bsz> trg sentences, each represented as a list of word indices 
# (including the sos and eos tokens)
def getBatch(batch_PairSentIndices, dev):

    bsz = len(batch_PairSentIndices)
    
    batch_srcSentIndices = []
    batch_trgSentIndices = []

    ### ???am i only zipping them up just to instantly unzip them again...???
    for pair in batch_PairSentIndices: # "unzip"
        batch_srcSentIndices.append(pair[0])
        batch_trgSentIndices.append(pair[1])

    encoder_inputs = batch_srcSentIndices
    decoder_inputs = []
    target_indices = []
    for j in range(bsz):
        decoder_inputs.append(batch_trgSentIndices[j][:-1]) # don't include <eos> token
        target_indices.append(batch_trgSentIndices[j][1:]) # don't include <sos> token

    src_lengths = torch.tensor([len(srcSent) for srcSent in encoder_inputs], device=dev).long()
    trg_lengths = torch.tensor([len(trgSent) for trgSent in target_indices], device=dev).long()

    # determine the length that each sentence of a batch must be padded to
    max_src_len = src_lengths.max()
    max_trg_len = trg_lengths.max()

    encoder_inputs_tensor = torch.zeros(bsz, max_src_len, device=dev).long() 
    decoder_inputs_tensor = torch.zeros(bsz, max_trg_len, device=dev).long() 
    target_indices_tensor = torch.zeros(bsz, max_trg_len, device=dev).long()
    for k in range(bsz):
        # kth sentence goes in kth row, which gets filled in up to the length of kth sentence
        # (contains 0 in every subsequent position of that row -> it is now padded)
        encoder_inputs_tensor[k,:src_lengths[k]] = torch.tensor(encoder_inputs[k]).long()
        decoder_inputs_tensor[k,:trg_lengths[k]] = torch.tensor(decoder_inputs[k]).long()
        target_indices_tensor[k,:trg_lengths[k]] = torch.tensor(target_indices[k]).long()

    print("encoder_inputs_tensor:")
    print(encoder_inputs_tensor)
    print()
    print("decoder_inputs_tensor:")
    print(decoder_inputs_tensor)
    print()
    print("target_indices_tensor:")
    print(target_indices_tensor)
    print()

    # (in DESCENDING order)
    sorted_src_lengths, idxs_in_enc_inputs_tensor = src_lengths.sort(0, descending=True)
    print("sorted_src_lengths:")
    print(sorted_src_lengths)
    print()
    print("idxs_in_enc_inputs_tensor:")
    print(idxs_in_enc_inputs_tensor)
    print()

    # -> use idxs_in_enc_inputs_tensor to sort encoder_inputs_tensor in descending order:
    # (only need it now, i.e., don't need to pass it to the encoder)
    encoder_inputs_tensor = encoder_inputs_tensor[idxs_in_enc_inputs_tensor]
    print("(sorted) encoder_inputs_tensor:")
    print(encoder_inputs_tensor)
    print()



    # -> rename src_lengths to correspond:
    ###!!!moved below
    #src_lengths = sorted_src_lengths

    # idxs_in_sorted_enc_inputs_tensor holds, at position i, the index within sorted_enc_inputs_tensor
    # of the ith tensor in encoder_inputs_tensor
    # (in ASCENDING order)
    _, idxs_in_sorted_enc_inputs_tensor = idxs_in_enc_inputs_tensor.sort(0)
    print("idxs_in_sorted_enc_inputs_tensor:")
    print(idxs_in_sorted_enc_inputs_tensor)
    print()
    # -> use it to unsort the resultant encodings back so that they correspond to the sorted decoder_inputs_tensor
    # (need to pass it to encoder)

    # construct mask to be used by attention mechanism of decoder
    ###???shouldnt this use the UNSORTED src lengths???
    mask = torch.ones(bsz, max_src_len, device=dev) == 1 # construct byte tensor
    for i in range(bsz):
        mask[i, :src_lengths[i]] = 0

    mask = mask.view(bsz, 1, max_src_len).expand(bsz, max_trg_len, max_src_len)
    
    ###!!!pass sorted_src_lengths, not renamed src_lengths
    encoder_inputs_batch = (encoder_inputs_tensor, sorted_src_lengths, idxs_in_sorted_enc_inputs_tensor)
    targets_batch = (target_indices_tensor, trg_lengths)

    decoder_inputs_batch = (decoder_inputs_tensor, trg_lengths, mask)

    return (encoder_inputs_batch, decoder_inputs_batch, targets_batch)



# no longer need to pack the decoder inputs, so sort by src
def getBatches(trainSentPairs, bsz, dev):
    start_time = time.time()
    batches = [] # list of triples, each corresponding to encoder inputs, decoder inputs, and targets for a given batch
    # sort everything ahead of time (by trg sentence length) for intelligent batching:
    
    print("sorting by trg length")
    trainSentPairs = sorted(trainSentPairs, key = lambda pair: len(pair[1]), reverse=True)
    #for i in range(len(trainSentPairs)):
    #    print("src {}, trg {}".format(len(trainSentPairs[i][0]), len(trainSentPairs[i][1])))
    #print()

    for i in range(0, len(trainSentPairs), bsz):
        batch_PairSentIndices = trainSentPairs[i:i+bsz]
        batches.append(getBatch(batch_PairSentIndices, dev))
    print("took %0.2f seconds to get all the batches" % (time.time()-start_time))
    return batches




# takes all the dev sentences and performs inference on batches of size bsz
def getDevBatch(sorted_triples_batch, dev):
    
    bsz = len(sorted_triples_batch) # number of sentences in this batch 
    #(NOT always equal to bsz within getDevBatches, bc last batch will be diff size if len(devset) not divisible by bsz)
    # 1D tensor that holds, at position i, the length of the ith srcSent of the batch
    src_lengths = torch.tensor([triple[1] for triple in sorted_triples_batch], device=dev).long()

    # so can unsort translations after inference
    corpus_indices = torch.tensor([triple[0] for triple in sorted_triples_batch], device=dev).long()

    max_src_len = src_lengths.max()
    encoder_inputs_tensor = torch.zeros(bsz, max_src_len, device=dev).long() 
    
    for k in range(bsz):
        encoder_inputs_tensor[k,:src_lengths[k]] = torch.tensor(sorted_triples_batch[k][2]).long()
        

    # perform so that meet encoder's spec
    sorted_src_lengths, idxs_in_enc_inputs_tensor = src_lengths.sort(0, descending=True)
    encoder_inputs_tensor = encoder_inputs_tensor[idxs_in_enc_inputs_tensor] 
    # -> rename src_lengths to correspond:
    src_lengths = sorted_src_lengths
    _, idxs_in_sorted_enc_inputs_tensor = idxs_in_enc_inputs_tensor.sort(0)


    # construct mask to be used by attention mechanism of decoder
    mask = torch.ones(bsz, max_src_len, device=dev) == 1 # construct byte tensor
    for i in range(bsz):
        mask[i, :src_lengths[i]] = 0
    mask = mask.view(bsz, 1, max_src_len) 

    encoder_inputs_batch = (encoder_inputs_tensor, src_lengths, idxs_in_sorted_enc_inputs_tensor)
    
    decoder_inputs_batch = (mask, corpus_indices)

    return (encoder_inputs_batch, decoder_inputs_batch)




def getDevBatches(src_sentences, bsz, dev):
    start_time = time.time()

    triples = [] # (index in orig corpus, src_length, src_sentence)
    for idx, src_sent in enumerate(src_sentences):
        triples.append((idx, len(src_sent), src_sent))

    # sort in desc order by src_length, keeping track of orig index in corpus
    sorted_triples = sorted(triples, key = lambda triple: triple[1], reverse=True)

    devBatches = [] # list of pairs, each corresponding to (encoder inputs, decoder inputs) for a given batch
    for i in range(0, len(src_sentences), bsz):
        sorted_triples_batch = sorted_triples[i:i+bsz]
        devBatches.append(getDevBatch(sorted_triples_batch, dev))
    print("took %0.2f seconds to get all the devBatches" % (time.time()-start_time))
    return devBatches

