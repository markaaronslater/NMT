import torch
from pickle import load

from src.model.lstm.model import EncoderDecoder

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
# checkpoint of a training session.
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
def load_checkpoint(hyperparams, checkpoint_path, name):
    checkpoint = torch.load(f"{checkpoint_path}{name}.tar") 
    model = initialize_model(hyperparams)
    optimizer = initialize_optimizer(model, hyperparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint


# load pretrained model so can perform inference.
def load_pretrained(checkpoint_path, name="best_model"):
    model_data = retrieve_model_data(checkpoint_path=checkpoint_path)
    model, _, _ = load_checkpoint(model_data["hyperparams"], checkpoint_path, name)
    model.decoder.set_inference_alg("beam_search")

    return model, model_data


def retrieve_model_data(checkpoint_path='/content/gdrive/My Drive/NMT/checkpoints/my_model/'):
    return load(open(f"{checkpoint_path}model_data.pkl", 'rb'))