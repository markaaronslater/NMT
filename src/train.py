import torch.nn as nn
import time, random

from predict import predict
from checkpoint import store_checkpoint, load_checkpoint


# train_batches holds training data as tensors.
# dev_batches holds dev data as tensors, for estimating model quality by its performance (bleu score) on the dev set.
# references used for calculating dev set bleu score.
# idx_to_trg_word holds dictionary that maps decoder predictions (which are indices in a vocabulary) to their corresponding words, for use during prediction of dev set.
# total_epochs is number of epochs to train the model across ALL sessions, not just this training session (e.g., if load a model to continue training from, which has already trained 5 epochs, and total_epochs = 10, then this session will only train for 5 more).
# folder is the name of the directory that holds checkpoints and translations folders.


# each checkpoint is on the order of ~1 GB, and most of them are not contenders (e.g., in early epochs, when weights not yet optimized), so only ever store the most recent model (e.g., so can resume training at a later time), and the best model found so far (so upon termination (by meeting early-stopping threshold or otherwise), can return the best model for later predicting the test set.
def train(translator, optimizer, train_batches, dev_batches, references, idx_to_trg_word, total_epochs=10, folder='./', threshold=5, from_scratch=True):
    ce_loss = torch.nn.CrossEntropyLoss()
    cur_epoch, best_bleu, prev_bleu, bad_epochs_count = 0, 0, 0, 0
    if not from_scratch:
        # resume training previous model checkpoint
        translator, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count = load_checkpoint(translator, optimizer, folder, "most_recent_model")
        cur_epoch = epoch + 1
        print(f"loaded model checkpoint from epoch: {epoch}, loss: {epoch_loss}, bleu: {bleu}, prev_bleu: {prev_bleu}, best_bleu: {best_bleu}, bad_epochs_count: {bad_epochs_count}")
        print(f"resuming training from epoch {cur_epoch}")
    
    for epoch in range(cur_epoch, total_epochs):
        epoch_loss = 0.
        epoch_start_time = time.time()
        random.shuffle(train_batches)
        for (encoder_inputs, decoder_inputs, decoder_targets) in train_batches:
            dists = translator(encoder_inputs, decoder_inputs)
            batch_loss = ce_loss(dists, decoder_targets)
            epoch_loss += batch_loss.detach()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        bleu, preds_time = predict(translator, dev_batches, references, idx_to_trg_word, folder, epoch, inference_alg="greedy_search", write=True)
        translator.train()
        report_stats(epoch, epoch_loss, epoch_time, preds_time, bleu, folder)
        # if this epoch model performed better on dev set than prev epoch model, bad_epochs_count resets to 0. (need not have outperformed best model, just the most recent model)
        bad_epochs_count = (bad_epochs_count+1) if epoch > 0 and bleu <= prev_bleu else 0
        if bleu > best_bleu:
            best_bleu = bleu
            store_checkpoint(translator, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, folder, "best_model") # store best checkpoint, so when early stopping terminates, can load it, rather than return current suboptimal checkpoint
        store_checkpoint(translator, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, folder, "most_recent_model") # always store most recent checkpoint, e.g., so can pick up training at later time
        if bad_epochs_count == threshold:
            # early-stopping threshold met
            best_translator = load_checkpoint(translator, optimizer, folder, "best_model")
            return best_translator
        prev_bleu = bleu

    best_translator = load_checkpoint(translator, optimizer, folder, "best_model")
    return best_translator


def report_stats(ep, ep_loss, ep_time, preds_time, bleu, folder):
    ep_stats = 'ep: {:02d}, loss: {:.8f}, ep_t: {:.2f} sec, t_t: {:.2f} sec, bleu: {:.4f}'.format(ep, ep_loss, ep_time, preds_time, bleu)
    print(ep_stats)
    with open(folder + 'model_train_stats.txt', 'a') as f:
        f.write(ep_stats)
        f.write('\n')