# each line of config_file is of the form:
# hyperparameter_name=hyperparameter_setting
# (see notebook or readme to observe supported hyperparameters and their names)
# ex)
# bidirectional_encoder=True
# hidden_layer_size=500
# ...

# call from Jupyter notebook, so path is:
# '/content/gdrive/My Drive/RNMT/configs'
def read_configs(*config_files):
    hyperparams = {}
    for config_file in config_files:
        read_config(config_file, hyperparams)
        
    return hyperparams


def read_config(config_file, hyperparams):
    with open(config_file, 'rt') as f:
        hp_name_val_pairs = f.read().strip().split('\n')
        for name_val in hp_name_val_pairs:
            name, val = name_val.strip().split('=')
            hyperparams[name] = val



        ### do this instead when initialize attributes of PyTorch models, so can decide on case by case basis what its data type ought to be
        # if val.isnumeric():
        #     val = float(val)
        ##!!!remember to handle ints separately, if necessary

    return hyperparams # for when not called by driver function read_configs


# project structure:
# /RNMT
#     /src/model/lstm/
#     /configs/
def instantiate_model(hyperparams, vocab_sizes):

    ### assertions
    if hyperparams['tie_weights']:
        assert int(hyperparams['hidden_size']) == int(hyperparams['input_size'])




    ### device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if hyperparams["device"] == 'cuda:0':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("warning: config file specified gpu, but only cpu available. using cpu...")
            device = "cpu"
    else:
        device = "cpu"


    encoder = Encoder(hyperparams, vocab_sizes["encoder"])
    decoder = Decoder(hyperparams, vocab_sizes["decoder"])
    model = EncoderDecoder(hyperparams, encoder, decoder)

    return model
