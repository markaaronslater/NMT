import torch
import torch.nn as nn
import time

from src.postprocessing.postprocess import postprocess

def predict(model, test_batches, idx_to_trg_word, folder='', ep='', write=True):
    with torch.no_grad():
        model.eval()
        model.encoder.eval()
        model.decoder.eval()
        preds_start_time = time.time()
        translation_idx_pairs = [(model(encoder_inputs, decoder_inputs), corpus_indices) for (encoder_inputs, decoder_inputs, corpus_indices) in test_batches]
        preds_time = time.time() - preds_start_time
    
        post_start_time = time.time()
        post_processed_translations = postprocess(translation_idx_pairs, idx_to_trg_word, model.decoder.eos_idx)
        post_time = time.time() - post_start_time

        if write:
            write_translations(post_processed_translations, model.decoder.inference_alg, folder, ep)

    return post_processed_translations, preds_time, post_time


def write_translations(translations, inference_alg, folder, ep):
    file_prefix = "beam_preds" if inference_alg == "beam_search" else "greedy_preds"
    with open(folder + file_prefix + str(ep) + '.txt', 'w') as f:
        for translation in translations:
            f.write(translation)
            f.write('\n')