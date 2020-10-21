# each line of config_file is of the form:
# hyperparameter_name=hyperparameter_setting
# (see notebook or readme to observe supported hyperparameters and their names)
# ex)
# bidirectional_encoder=True
# hidden_layer_size=500
# ...
def get_params(config_file):
    with open(config_file, 'rt', encoding='utf-8') as f:
        hp_name_val_pairs = f.read().strip().split('\n')
        hyperparams = {}
        for name_val in hp_name_val_pairs:
            name, val = name_val.split('=')
            # numeric values must be type-converted
            if val.isnumeric():
                val = float(val)
            # !!!handle ints and floats separately
            hyperparams[name] = val
    return hyperparams

def instantiate_model(hyperparams):