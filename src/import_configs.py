# each line of config_file is of the form:
# hyperparameter_name=hyperparameter_setting
# (see notebook or readme for supported hyperparameters and their names)
# ex)
# bidirectional_encoder=True
# enc_hidden_size=500
# ...
def import_configs(config_path='/content/gdrive/My Drive/NMT/configs/', unittesting=False):
    hyperparams = {}
    read_configs(hyperparams, config_path)
    process_configs(hyperparams)
    if unittesting:
        overwrite_configs(hyperparams)
    constrain_configs(hyperparams)

    return hyperparams


def read_configs(hyperparams, config_path='/content/gdrive/My Drive/NMT/configs/'):
    #!!!TODO: make so that all stored in single config file
    for config_file in ["encoder.txt", "decoder.txt", "train.txt", "vocab.txt"]:
        read_config(hyperparams, config_path + config_file)


def read_config(hyperparams, config_file):
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
    assert hyperparams["optimization_alg"] in ["Adam", "AdamW"]
    assert hyperparams["decoder_init_scheme"] in ["layer_to_layer", "final_to_first"]
    assert hyperparams["attention_fn"] in ["dot_product", "scaled_dot_product", "none"] 
    assert hyperparams["vocab_type"] in ["subword_ind", "subword_joint", "subword_pos", "word"]
    assert hyperparams["trim_type"] in ["threshold", "top_k"]

    # ensure every setting provided for integer-valued hyperparams is castable as int.
    int_hyperparams =   [   "train_bsz", "dev_bsz", "test_bsz", 
                            "enc_input_size", "enc_hidden_size", "enc_num_layers",
                            "dec_input_size", "dec_hidden_size", "dec_num_layers",
                            "beam_width", "decode_slack", 
                            "src_k", "trg_k", "src_thres", "trg_thres",
                            "vocab_threshold", "num_merge_ops"]
    for hp in int_hyperparams:
        if hyperparams[hp].isdigit():
            hyperparams[hp] = int(hyperparams[hp])
        else:
            raise ValueError(f"error: provided a non-integer value for {hp}: {hyperparams[hp]}. see readme for proper input formats.")

    bool_hyperparams = [    "bidirectional", "project",
                            "reverse_src", "tie_weights", "attention_layer"]
    for hp in bool_hyperparams:
        if hyperparams[hp] in ["True", "False"]:
            hyperparams[hp] = True if hyperparams[hp] == "True" else False
        else:
            raise ValueError(f"error: provided a non-boolean value for {hp}: {hyperparams[hp]}. see readme for proper input formats.")

    float_hyperparams = [   "learning_rate", "L2_reg", "enc_dropout", "enc_lstm_dropout", 
                            "dec_dropout", "dec_lstm_dropout"]
    for hp in float_hyperparams:
        # will raise ValueError on its own if passed non-numeric value.
        hyperparams[hp] = float(hyperparams[hp])


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
        if hyperparams["attention_fn"] != "none":
            # -if use attention, must apply attention layer to project
            # attentional states to input_size.
            assert hyperparams["attention_layer"]
        else:
            # -if dn use attention, then input_size must equal hidden_size.
            assert hyperparams["dec_hidden_size"] == hyperparams["dec_input_size"]

    # lstm dropout parameter only applies to non-final lstm layers
    if hyperparams['enc_num_layers'] == 1:
        assert hyperparams['enc_lstm_dropout'] == 0.0
    if hyperparams['dec_num_layers'] == 1:
        assert hyperparams['dec_lstm_dropout'] == 0.0 

    # if joint vocab, then share an embeddings table
    if hyperparams["vocab_type"] == "subword_joint":
        assert hyperparams["enc_input_size"] == hyperparams["dec_input_size"]


# use mostly the same default hyperparams as when actually training, but overwrite
# such that applies to toy dataset (e.g., first 10 sentences of training sets).
def overwrite_configs(hyperparams):
    hyperparams["train_bsz"] = 3
    hyperparams["dev_bsz"] = 3
    hyperparams["decode_slack"] = 30 # set large enough such that can finish predicting each of the 10 target sentences (or else will not achieve BLEU of 100)
    hyperparams["enc_hidden_size"] = 1500 # ensure model is of sufficient capacity
    hyperparams["dec_hidden_size"] = 1500
    hyperparams["enc_dropout"] = 0. # ensure regularization turned off
    hyperparams["dec_dropout"] = 0.
    hyperparams["enc_lstm_dropout"] = 0.
    hyperparams["dec_lstm_dropout"] = 0.
    hyperparams["L2_reg"] = 0.
    hyperparams["src_k"] = 200 # set large enough such that every word included in vocab (or else will not achieve BLEU of 100)
    hyperparams["trg_k"] = 200
    hyperparams["src_thres"] = 1 # every word included in vocab
    hyperparams["trg_thres"] = 1
    hyperparams["num_merge_ops"] = 300    
    hyperparams["vocab_threshold"] = 0 # every word included in vocab