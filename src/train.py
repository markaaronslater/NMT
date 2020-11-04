import torch
import torch.nn as nn
import time, random

from src.predict import predict
from src.evaluate import evaluate
from src.preprocessing.preprocess import retrieve_model_data
from src.preprocessing.corpus_utils import get_references
from src.model.lstm.model import EncoderDecoder

# -train_batches holds training data as tensors.
# -dev_batches holds dev data as tensors, for estimating model quality
# by its performance (bleu score) on the dev set.
# -references used for calculating dev set bleu score.
# -idx_to_trg_word holds dictionary that maps decoder predictions
# (which are indices in a vocabulary) to their corresponding words,
# for use during prediction of dev set.
# -total_epochs is number of epochs to train the model across ALL sessions,
# not just this training session (e.g., if load a model to continue training
# from, which has already trained 5 epochs, and total_epochs = 10,
# then this session will only train for 5 more).
# -checkpoint_path denotes location where will store:
#   -checkpoints,
#   -epoch stats for each checkpoint, and
#   -text files holding greedy predictions of each checkpoint.
#   -if from_scratch==False, then checkpoint_path holds path to checkpoint to load.
# -save denotes whether or not to save checkpoints (e.g., do not save
# when running particular unit tests).

# each checkpoint is on the order of ~1 GB, and most of them are not contenders
# (e.g., in early epochs, when weights not yet optimized), so only ever store
# the most recent model, e.g., so can resume training at a later time,
# and the best model found so far, so upon termination
# (by meeting early-stopping threshold or otherwise),
# can return the best model for later predicting the test set.

# train a model on preprocessed data inside checkpoint path, of hyperparameters inside checkpoint path.
def train(checkpoint_path='/content/gdrive/My Drive/NMT/checkpoints/my_model/',
        from_scratch=True, save=True, overfit=False):
    ### instantiate model ###
    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    train_batches = model_data["train_batches"]
    dev_batches = model_data["dev_batches"]
    idx_to_trg_word = model_data["idx_to_trg_word"]
    hyperparams = model_data["hyperparams"]
    dev_references = get_references(overfit=overfit)
    model = initialize_model(hyperparams)
    optimizer = initialize_optimizer(model, hyperparams)

    ce_loss = torch.nn.CrossEntropyLoss()
    total_epochs = hyperparams["total_epochs"]
    early_stopping = hyperparams["early_stopping"]
    threshold = hyperparams["early_stopping_threshold"]

    
    if from_scratch:
        start_epoch, best_bleu, prev_bleu, bad_epochs_count = 1, 0, 0, 0
    else:
        # resume training previous model checkpoint
        model, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count = load_checkpoint(model, optimizer, checkpoint_path, "most_recent_model")
        start_epoch = epoch + 1
        if start_epoch == 1:
            print(f"training model from scratch")
        else:
            print(f"loaded model checkpoint from epoch: {epoch}")
            print(f"loss: {epoch_loss}, bleu: {bleu}, prev_bleu: {prev_bleu}, \
                    best_bleu: {best_bleu}, bad_epochs_count: {bad_epochs_count}")
            print(f"resuming training from epoch {start_epoch}")
            print()



    ### training loop ###
    for epoch in range(start_epoch, total_epochs+1):
        epoch_loss = 0.
        epoch_start_time = time.time()
        random.shuffle(train_batches)
        for batch in train_batches:
            epoch_loss += training_step(model, optimizer, ce_loss, batch)
            
        epoch_time = time.time() - epoch_start_time
        dev_translations, preds_time, post_time = predict(model, dev_batches, idx_to_trg_word, checkpoint_path, epoch, write=True)
        bleu = evaluate(dev_translations, dev_references)
        
        model.train()
        model.encoder.train()
        model.decoder.train()
        report_stats(epoch, epoch_loss, epoch_time, preds_time, bleu, checkpoint_path, post_time)

        if early_stopping:
            # if this epoch model performed better on dev set than prev epoch model,
            # bad_epochs_count resets to 0. (need not have outperformed best model,
            # just the most recent model).
            bad_epochs_count = (bad_epochs_count+1) if epoch > 0 and bleu <= prev_bleu else 0
            if bleu > best_bleu:
                best_bleu = bleu
                # when terminates, can load best model, rather than current suboptimal checkpoint.
                store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, checkpoint_path, "best_model")
            if bad_epochs_count == threshold:
                # early-stopping threshold met
                best_model, epoch_loss = load_checkpoint(model, optimizer, checkpoint_path, "best_model")
                return best_model, epoch_loss

        if save:
            # store checkpoint each epoch, e.g., so can pick up training at later time.
            store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, checkpoint_path, "most_recent_model")
        prev_bleu = bleu

    if early_stopping:
        best_model = load_checkpoint(model, optimizer, checkpoint_path, "best_model")
        return best_model, epoch_loss
    else:
        return model, epoch_loss



def training_step(model, optimizer, loss, batch):
    encoder_inputs, decoder_inputs, decoder_targets = batch
    dists = model(encoder_inputs, decoder_inputs)
    batch_loss = loss(dists, decoder_targets)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.detach()


def report_stats(ep, ep_loss, ep_time, preds_time, bleu, checkpoint_path, post_time):
    ep_stats = 'ep: {:02d}, loss: {:.8f}, ep_t: {:.2f} sec, t_t: {:.2f} sec, p_t: {:.2f} sec, bleu: {:.4f}'.format(ep, ep_loss, ep_time, preds_time, post_time, bleu)
    print(ep_stats)
    with open(checkpoint_path + 'model_train_stats.txt', 'a') as f:
        f.write(ep_stats)
        f.write('\n')


def initialize_model(hyperparams):
    model = EncoderDecoder(hyperparams)
    if hyperparams["device"] == "cuda:0" and torch.cuda.is_available():
        model.cuda()
    elif hyperparams["device"] == "cuda:0":
        print(f"warning: config specified cuda:0 but only a cpu is available. \
                must change device setting to 'cpu', and re-call \
                retrieve_model_data() before call train().")

    return model


def initialize_optimizer(model, hyperparams):
    opt_alg = hyperparams["optimization_alg"]
    lr = hyperparams["learning_rate"]
    wd = hyperparams["L2_reg"]
    if opt_alg == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_alg == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    return optimizer




# name in ["best_model", "most_recent_model"]
def store_checkpoint(model, optimizer, epoch, epoch_loss, bleu, prev_bleu,
            best_bleu, bad_epochs_count, checkpoint_path, name):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        'bleu': bleu,
        'prev_bleu':prev_bleu,
        'best_bleu':best_bleu,
        'bad_epochs_count':bad_epochs_count
    }, f"{checkpoint_path}{name}.tar")


# name in ["best_model", "most_recent_model"]
def load_checkpoint(model, optimizer, checkpoint_path, name):
    checkpoint = torch.load(f"{checkpoint_path}{name}.tar") 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # if loading most_recent_model, then resuming training,
    # so need to return all associated epoch information,
    # but if loading best_model, finished training and now want to
    # predict test set, so just return the model and its loss.
    if name == "most_recent_model":
        return model, optimizer, checkpoint['epoch'], \
        checkpoint['epoch_loss'], checkpoint['bleu'], \
        checkpoint['prev_bleu'], checkpoint['best_bleu'], \
        checkpoint['bad_epochs_count']
    else:
        return model, checkpoint['epoch_loss']