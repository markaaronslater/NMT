import torch
import torch.nn as nn
import time
from nltk.translate.bleu_score import corpus_bleu

# this same function is also used for predicting the test set.
# use greedy search if measuring model performance on dev set during training ( cheaper, and beam search will always perform at least as well anyway).
# use "beam_search" if measuring model performance on test set after found a best model during training.
def predict(translator, dev_batches, references, idx_to_trg_word, folder, ep, inference_alg="greedy_search", write=True):
    # get predicted translations
    with torch.no_grad():
        translator.eval()
        translator.decoder.inference_alg = inference_alg
        preds_start_time = time.time()

        translation_idx_pairs = [(translator(encoder_inputs, decoder_inputs), corpus_indices) for (encoder_inputs, decoder_inputs, corpus_indices) in dev_batches]

        preds_time = time.time() - preds_start_time
    
        

        # post-process the translations
        post_processed_translations = post_process(translation_idx_pairs, idx_to_trg_word, int(translator.decoder.eos_idx))

        # write to file
        if write:
            write_translations(post_processed_translations, translator.decoder.inference_alg, folder, ep)

        # estimate performance on dev set, or determine performance on test set with NLTK
        hypotheses = [translation.split() for translation in post_processed_translations]
        bleu = corpus_bleu(references, hypotheses)

    return bleu, preds_time


def write_translations(translations, inference_alg, folder, ep):
    file_prefix = "beam_preds" if inference_alg == "beam_search" else "greedy_preds"
    with open(folder + file_prefix + str(ep) + '.txt', 'w') as f:
        for _, translation in translations:
            f.write(translation)
            f.write('\n')