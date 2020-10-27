import torch

# name in ["best_model", "most_recent_model"]
def store_checkpoint(translator, optimizer, ep, ep_loss, bleu, prev_bleu, best_bleu, bad_epochs_count, folder, name):
    torch.save({
        'model_state_dict': translator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': ep,
        'epoch_loss': ep_loss,
        'bleu': bleu,
        'prev_bleu':prev_bleu,
        'best_bleu':best_bleu,
        'bad_epochs_count':bad_epochs_count
    }, folder + name + '.tar')


def load_checkpoint(translator, optimizer, folder, name):
    checkpoint = torch.load(folder + name + '.tar') 
    translator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # if loading most_recent_model, then resuming training, so need to return all associated epoch information, but if loading best_model, finished training and now want to predict test set, so just return the translator
    if name == "most_recent_model":
        return translator, optimizer, checkpoint['epoch'], checkpoint['epoch_loss'], checkpoint['bleu'], checkpoint['prev_bleu'], checkpoint['best_bleu'], checkpoint['bad_epochs_count']
    else:
        return translator