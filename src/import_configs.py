# each line of config_file is of the form:
# hyperparameter_name=hyperparameter_setting
# (see notebook or readme for supported hyperparameters and their names)
# ex)
# bidirectional_encoder=True
# enc_hidden_size=500
# ...
def read_configs(config_path='/content/gdrive/My Drive/NMT/configs/'):
    #!!!TODO: make so that all stored in single config file
    config_files = ["encoder.txt", "decoder.txt", "train.txt", "vocab.txt"]

    hyperparams = {}
    for config_file in config_files:
        read_config(config_path + config_file, hyperparams)
        
    process_configs(hyperparams)
    constrain_configs(hyperparams)

    return hyperparams


def read_config(config_file, hyperparams):
    with open(config_file, 'rt') as f:
        hp_name_val_pairs = f.read().strip().split('\n')
        for name_val in hp_name_val_pairs:
            name, val = name_val.strip().split('=')
            hyperparams[name] = val


# 1-ensure supplied valid values, 
# 2-perform relevant type conversions.
def process_configs(hyperparams):
    # string-type hyperparams
    assert hyperparams["device"] in ["cuda:0", "cpu"]
    assert hyperparams["attention_fn"] in ["dot_product", "scaled_dot_product", "none"] 
    assert hyperparams["inference_alg"] in ["beam_search", "greedy_search"]
    assert hyperparams["vocab_type"] in ["subword_ind", "subword_joint", "subword_pos", "word"]
    assert hyperparams["trim_type"] in ["threshold", "top_k"]
    assert hyperparams["optimizer"] in ["Adam", "AdamW"]
    assert hyperparams["decoder_init_scheme"] in ["layer_to_layer", "final_to_first"]

    # ensure every setting provided for integer-valued hyperparams is castable as int.
    int_hyperparams =   [   "num_epochs", "train_bsz", "dev_bsz", "test_bsz", 
                            "enc_input_size", "enc_hidden_size", "enc_num_layers",
                            "dec_input_size", "dec_hidden_size", "dec_num_layers",
                            "beam_width", "decode_slack", 
                            "src_k", "trg_k", "src_thres", "trg_thres"]
    for hp in int_hyperparams:
        if hyperparams[hp].isdigit():
            hyperparams[hp] = int(hyperparams[hp])
        else:
            raise ValueError(f"error: provided a non-integer value for {hp}: {hyperparams[hp]}. see readme for proper input formats.")

    bool_hyperparams = [    "early_stopping", "bidirectional", "project",
                            "reverse_src", "tie_weights", "attention_layer"]
    for hp in bool_hyperparams:
        if hyperparams[hp] in ["True", "False"]:
            hyperparams[hp] = bool(hyperparams[hp])
        else:
            raise ValueError(f"error: provided a non-boolean value for {hp}: {hyperparams[hp]}. see readme for proper input formats.")

    float_hyperparams = ["learning_rate", "L2_reg", "enc_dropout", "dec_dropout"]
    for hp in float_hyperparams:
        # will raise ValueError on its own if passed non-numeric value.
        hyperparams[hp] = float(hyperparams[hp])
        #raise ValueError(f"error: provided a value for {hp} that cannot cast to float: {hyperparams[hp]}. see readme for proper input formats.")


# ensure passed valid combinations of configs
def constrain_configs(hyperparams):
    if hyperparams["bidirectional"] and not hyperparams["project"]:
        # attention fn forms dot product on concatenated encoder states,
        # so decoder state must be twice their dimensionality
        assert hyperparams["dec_hidden_size"] == 2 * hyperparams["enc_hidden_size"]
    else:    
        assert hyperparams["dec_hidden_size"] == hyperparams["enc_hidden_size"]

    if hyperparams["decoder_init_scheme"] == "layer_to_layer":
        assert hyperparams["enc_num_layers"] == hyperparams["dec_num_layers"]

    # only project hidden state with intermediate layer if use attention mechanism
    assert (hyperparams["attention_fn"] != "none" or not hyperparams["attention_layer"])

    if hyperparams['tie_weights']:
        assert hyperparams['dec_input_size'] == hyperparams['dec_hidden_size']