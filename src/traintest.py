import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from random import shuffle
from nltk.translate.bleu_score import corpus_bleu

import sys

# modified version of train that saves model after each epoch
# further, we now pass it an optimizer rather than initializing inside, so can continue training a checkpointed model
# folder is the name of the directory that holds checkpoints and translations folders
def train(translator, optimizer, trainBatches, devBatches, references, num_epochs=10, cur_ep=0, folder='./', save=True):
    nll = torch.nn.NLLLoss()
    for ep in range(cur_ep, num_epochs):
        ep_loss = 0.
        ep_start_time = time.time()
        shuffle(trainBatches)
        for (encoder_inputs_batch, decoder_inputs_batch, targets_batch) in trainBatches:
            packedDists = translator(encoder_inputs_batch, decoder_inputs_batch)
            ###!!! new spec
            #packedTargets, _ = pack_padded_sequence(targets_batch[0], targets_batch[1], batch_first=True)
            #??why didnt i do this ahead of time??
            packedTargets = pack_padded_sequence(targets_batch[0], targets_batch[1], batch_first=True)
            packedTargets = packedTargets.data
            batch_loss = nll(packedDists, packedTargets)
            ep_loss += batch_loss.detach()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        # save model checkpoint after each epoch ###
        ###!!!place this in a helper function. put all relevant data in a dictionary so can pass with single param, etc.
        if save:
            torch.save({
                'epoch': ep,
                'model_state_dict': translator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ep_loss': ep_loss,
            }, folder + 'cp' + str(ep) + '.tar')

        ep_time = time.time()-ep_start_time


        # does this help??
        del packedDists, packedTargets

        # get OOS error on dev set:
        with torch.no_grad():
            #test_time, nodrop_bleu, bleu_time = test(translator, devBatches, references, folder, ep, write=False)
            translator.eval()
            test_time, drop_bleu, bleu_time = test(translator, devBatches, references, folder, ep, write=True)
            translator.train()

        translator.decoder.tf = True

        #ep_stats = 'ep: {:02d}, loss: {:.2f}, ep_t: {:.2f} sec, t_t: {:.2f} sec, bleu1: {:.4f}, bleu2: {:.4f}'.format(ep, ep_loss, ep_time, test_time, nodrop_bleu, drop_bleu)
        ep_stats = 'ep: {:02d}, loss: {:.8f}, ep_t: {:.2f} sec, t_t: {:.2f} sec, bleu: {:.4f}'.format(ep, ep_loss, ep_time, test_time, drop_bleu)

        print(ep_stats)
        with open(folder + 'model_train_stats.txt', 'a') as f:
            f.write(ep_stats + "\n")



        # with open(folder + 'preds' + str(ep) + '.txt', 'w') as f:
        #     for translation in translations:
        #         f.write(translation + '\n')


    return translator, ep_loss

















def test(translator, devBatches, references, folder, ep, write=True):
    # output words based on previously generated words,
    # instead of outputting distsOverNextWords based on previous target words
    test_start_time = time.time()
    translator.decoder.tf = False
    #translator.eval()

    corpus_translations = [] # list of (corpus_idx, str(translation)) pairs
    for (encoder_inputs_batch, decoder_inputs_batch) in devBatches:
        translations_batch = translator(encoder_inputs_batch, decoder_inputs_batch) 
        #for translation in translations:
        #    print(translation) # in case of file writing failures, extract from jupyter display
        #list_of_translations.append(translations)
        corpus_translations += translations_batch
        

    # unsort for comparison with dev trg sentences
    unsorted_corpus_translations = sorted(corpus_translations, key = lambda pair: pair[0])
    del corpus_translations
    #trans = [] # unsorted list of translations without idxs
    hypotheses = [] # for NLTK bleu computation

    if write:
        filePrefix = 'preds'
        if translator.decoder.inf_alg == "beam_search":
            filePrefix = 'beam' + filePrefix
        with open(folder + filePrefix + str(ep) + '.txt', 'w') as f:
            for idx, translation in unsorted_corpus_translations:
                f.write(translation + '\n')
                #trans.append(translation)
                hypotheses.append(translation.split())

    else:
        for idx, translation in unsorted_corpus_translations:
            hypotheses.append(translation.split())

    # may show higher value now, bc includes the write operations
    test_time = time.time() - test_start_time
    #print("{} decoding took {} seconds".format(translator.decoder.alg, test_time))

    # estimate out-of-sample error on dev set with nltk
    bleu_start_time = time.time()
    # print("references:")
    # print(references)
    # print("hypotheses:")
    # print(hypotheses)
    # print()
    bleu = corpus_bleu(references, hypotheses)
    bleu_time = time.time() - bleu_start_time
    #!!!must reset translator properties back to training versions, i think
    translator.decoder.tf = True
    #translator.train()

    #return trans, test_time, bleu, bleu_time
    return test_time, bleu, bleu_time






def trainAndTest():
    pass


