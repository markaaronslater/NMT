# each line of config_file is of the form:
# hyperparameter_name=hyperparameter_setting
# (see notebook or readme for supported hyperparameters and their names)
# ex)
# bidirectional_encoder=True
# hidden_layer_size=500
# ...

# called from Jupyter notebook, so path is:
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