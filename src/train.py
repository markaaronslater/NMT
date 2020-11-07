import torch
import torch.nn as nn
import torch.nn.functional as F
import time, random

from src.predict import predict
from src.evaluate import evaluate
from src.model_utils import load_checkpoint, store_checkpoint, retrieve_model_data

# each checkpoint is on the order of ~1 GB, and most of them are not contenders
# (e.g., in early epochs, when weights not yet optimized), so only ever store
# the most recent model, e.g., so can resume training at a later time,
# and the best model found so far, so upon termination
# (by meeting early-stopping threshold or otherwise),
# can return the best model for later predicting the test set.

# train a model on preprocessed data inside checkpoint path, of hyperparameters inside checkpoint path.
def train(total_epochs=30, early_stopping=True, threshold=5,
            checkpoint_path='/content/gdrive/My Drive/NMT/checkpoints/my_model/',
            save=True, write=True):
    ### immutable training session data ###
    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    train_batches = model_data["train_batches"]
    dev_batches = model_data["dev_batches"]
    dev_references = model_data["references"]
    idx_to_trg_word = model_data["idx_to_trg_word"]
    hyperparams = model_data["hyperparams"]
    #######################################

    ### mutable training session data ###
    model, optimizer, checkpoint = load_checkpoint(hyperparams, checkpoint_path, "most_recent_model")
    epoch = checkpoint["epoch"]
    epoch_loss = checkpoint["epoch_loss"]
    bleu = checkpoint["bleu"]
    prev_bleu = checkpoint["prev_bleu"]
    best_bleu = checkpoint["best_bleu"]
    bad_epochs_count = checkpoint["bad_epochs_count"]
    #######################################    
    
    if epoch == 0:
        # loaded a checkpoint that has been trained for zero epochs.
        print("training model from scratch...")
        print()
        start_epoch = 1
    else:
        print(f"loaded model checkpoint from epoch: {epoch:02d}")
        print(f"loss: {epoch_loss:.4f}, bleu: {bleu:.2f}, prev_bleu: {prev_bleu:.2f}, best_bleu: {best_bleu:.2f}, bad_epochs_count: {bad_epochs_count:02d}")
        start_epoch = epoch + 1
        print(f"resuming training from epoch {start_epoch}...")
        print()

    ### training loop ##############################
    for epoch in range(start_epoch, total_epochs+1):
        epoch_loss = 0.
        random.shuffle(train_batches)
        epoch_start_time = time.time()
        for batch in train_batches:
            epoch_loss += training_step(model, optimizer, batch)
        epoch_time = time.time() - epoch_start_time

        dev_translations, preds_time, post_time = predict(model, dev_batches, idx_to_trg_word, checkpoint_path, epoch, write=write)
        bleu = evaluate(dev_translations, dev_references)
        
        model.train()
        model.encoder.train()
        model.decoder.train()
        report_stats(epoch, epoch_loss, epoch_time, preds_time, bleu, checkpoint_path, post_time)

        if early_stopping:
            # if this epoch model performed better on dev set than prev epoch model,
            # bad_epochs_count resets to 0. (need not have outperformed best model,
            # just the most recent model).
            bad_epochs_count = (bad_epochs_count+1) if epoch > 1 and bleu <= prev_bleu else 0
            if bleu > best_bleu:
                best_bleu = bleu
                # when terminates, can load best model, rather than potentially suboptimal model of final epoch.
                store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, checkpoint_path, "best_model")
            if bad_epochs_count == threshold:
                # early-stopping threshold met
                best_model, optimizer, checkpoint = load_checkpoint(hyperparams, checkpoint_path, "best_model")
                return best_model, checkpoint["epoch_loss"]

        if save:
            # store checkpoint each epoch, e.g., so can pick up training at later time.
            store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, checkpoint_path, "most_recent_model")
        
        prev_bleu = bleu
    ################################################

    if early_stopping:
        best_model, optimizer, checkpoint = load_checkpoint(hyperparams, checkpoint_path, "best_model")
        return best_model, checkpoint["epoch_loss"]
    else:
        return model, epoch_loss



def training_step(model, optimizer, batch):
    encoder_inputs, decoder_inputs, decoder_targets = batch
    dists = model(encoder_inputs, decoder_inputs)
    batch_loss = F.cross_entropy(dists, decoder_targets)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.detach()


def report_stats(ep, ep_loss, ep_time, preds_time, bleu, checkpoint_path, post_time):
    ep_stats = 'ep: {:02d}, loss: {:.4f}, ep_t: {:.2f} sec, t_t: {:.2f} sec, p_t: {:.2f} sec, bleu: {:.2f}'.format(ep, ep_loss, ep_time, preds_time, post_time, bleu)
    print(ep_stats)
    with open(checkpoint_path + 'model_train_stats.txt', 'a') as f:
        f.write(ep_stats)
        f.write('\n')