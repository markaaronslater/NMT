PyTorch implementation of an encoder-decoder LSTM (Long short-term memory) neural network that translates German to English.

coming soon:
-support for transformer architecture
-support for more language pairs and translation directions


model features:
-fully-parallelized (optimally efficient, i.e., no unnecessary loops in training and inference code).
-attention mechanism.
-both beam search and greedy search inference algorithms.
-extraction and application of either word-level or subword-level model vocabularies.
-high degree of model flexibility controlled with set of configuration files that set hyperparameters, architectural variations, etc.
-training algorithm performs early-stopping and, when run in Google Colab notebook, saves both most-recent (so can resume training at later date) and best-so-far (so can extract for later predicting test set, etc.) model checkpoints in Google Drive.
-all preprocessing performed starting from parallel set of train files, train.de and train.en, where line i of train.en is the ground-truth English translation of line i of train.de. Calls a series of pre-trained Stanford CoreNLP (Stanza) processors that perform tokenization, multi-word token expansion, and part-of-speech tagging on each corpus. This part-of-speech information is then leveraged by truecasing algorithm that employs linguistic heuristics to decide whether or not to convert a word to lowercase (so that a word's meaning is not distributed across capitalized and lower-case embeddings, size of vocabulary is reduced, etc.). These truecased corpuses of words are then optionally segmented into corpuses of subwords (employing byte-pair-encodings of https://github.com/rsennrich/subword-nmt-nmt). Lastly, intelligent batching is performed so that the number of pad tokens (in batches of variable-length sequences) is minimized during both training and inference, and so that PackedSequence objects can be used. coming soon: additional preprocessing steps, such as compound-splitting and punctuation-collapsing.


        possible configuration settings:
        (do not worry about passing an invalid combination of settings, bc constrain_configs() will raise an assertion error for you).
        (default settings shown after '=', and all possible options in [...] inside comments. if omitted, then if default is an int or boolean, must set as an int or boolean, respectively, and if default is a float, can set as int or float).

        
Encoder Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
bidirectional | True | boolean | if true, use bidirectional instead of unidirectional encoder
project | True | boolean | whether to project the concatenated forward and backward encoder states back to original dimensionality (so that same size as decoder hidden states when computing attention) (only used if bidirectional == True)
reverse_src | False | boolean |  whether to pass src sentence to encoder in reverse order (only used if bidirectional == False)
decoder_init_scheme | layer_to_layer | [layer_to_layer, final_to_first] | whether to have each layer of encoder directly initialize corresponding layer of decoder (requires same number of layers in each) or to have final layer of encoder initialize first layer of decoder (with all non-first decoder layers initialized as zeros)
enc_input_size | 300 | int | embedding size
enc_hidden_size | 600 | int |
enc_num_layers | 1 | int | 
enc_dropout | 0.0 | float | 


Decoder Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
tie_weights | False | boolean | whether or not to share weights in the embeddings matrix and the output matrix (requires input and hidden sizes to have same dimensionality)
attention_fn | dot_product | [dot_product, scaled_dot_product, none] | whether or not to include attention mechanism, and if so, which one
attention_layer | False | boolean | whether or not to project the attentional representations (which are contexts concatenated to hidden states) from 2*hidden_size back to hidden_size (along with a nonlinearity) before projecting them to vocab_size in output layer. only used if attention_fn is not "none". must be set to True if tie_weights == True, bc in this case hidden_size == input_size
beam_width | 10 | int | how many candidate translations to consider the running probabilities of (for a single src sentence) at any given moment during beam search inference
decode_slack | 20 | int | max number of words to predict beyond the length of corresponding source sentence length, before "times out" (controls strength of heuristic that an English translation will be a similar number of words as German source sentence) (cuts off translations that never produce <end-of-sentence> symbol)
dec_input_size | 300 | int | embedding size
dec_hidden_size | 600 | int | 
dec_num_layers | 1 | int |  
dec_dropout | 0.0 | float | 


Training Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
early_stopping | True | boolean | whether or not to quit training early if bleu scores on dev set do not improve for <early_stopping_threshold> epochs in a row (see below)
early_stopping_threshold | 5 | int | 
from_scratch | True | boolean | if False, then loads and resumes training a partially trained model checkpoint inside checkpoints
total_epochs | 30 | int | max number of epochs (can cut short if early-stopping)
learning_rate | .001 | float | 
L2_reg | 0.0 | float | weight decay of optimizer
optimization_alg | Adam | [Adam, AdamW] | 
device | cuda:0 | [cuda:0, cpu] |
train_bsz | 64 | int |
dev_bsz | 32 | int | 
test_bsz | 16 | int | 



Vocabulary Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
vocab_type | subword_joint | [subword_ind, subword_joint, word] | whether to learn independent subword vocabularies, a joint subword vocabulary, or independent word vocabularies, for source and target languages.
trim_type | threshold | [threshold, top_k] | only used if vocab_type == word
src_k | 30000 | int | integer denoting the number of words in src training corpus to retain in src vocab (only used if trim_type == top_k)
trg_k | 30000 | int | (only used if trim_type == top_k)
src_thres | 2 | int | integer denoting the minimum number of times a word must occur in src training corpus in order to be included in src vocab (only used if trim_type == threshold)
trg_thres | 2 | int | (only used if trim_type == threshold)
num_merge_ops | 30000 | int | used by bpe-encodings algorithm for setting subword vocab size (only used if vocab_type != word)
vocab_threshold | 10 | int | how many times subword must occur in corpus in order to not "de-merge" it back into smaller subwords (for handling out-of-vocabulary words at test time) (only used if vocab_type == subword_joint)



















playground.ipynb
follow notebook instructions to:
-observe translations of sample German sentences into English.
-interactively input German sentences to observe translation output, and coming soon: compare translation side-by-side with Google translate's output.



NMT-driver.ipynb
follow notebook instructions to:
a) replicate any subset of the 6 steps (see below), each in its own cell, needed to preprocess the data, train, and evaluate a model,
b) run unit tests to show correctness of model implementations, or
c) load a pre-trained model checkpoint to determine BLEU score on test set.

each preprocessing step creates a "checkpoint" of the preprocessed corpuses stored in files named with a corresponding prefix inside of <corpus_path> (default corpus path is /content/gdrive/My Drive/NMT/corpuses/iwslt16_en_de/), so that if wish to change a step, do not need to re-run all prior steps. preprocessed corpuses of any step can be retrieved via read_tokenized_corpuses() function.

step 1 - apply Stanza processors, which perform tokenization, multi-word token expansion, and part-of-speech tagging
processor outputs are saved to files with the prefix "stanza_"
ex) stanza output for train.en is stored in
/content/gdrive/My Drive/iwslt16_en_de/stanza_train.en.pkl


step 2 - truecase the corpuses using linguistic heuristics that leverage morphological data produced by morphological data tagger
truecased corpuses are saved to files with the prefix "word_" (bc they are used directly by models that employ a word-level vocabulary).

step 3 - segment corpuses of words into corpuses of subwords
segmented corpuses are saved to files with the prefix 
"subword_joint_" or "subword_ind_", depending on if learn a joint vocabulary or separate, independent vocabularies, respectively, for the source and target languages.

step 4 - convert corpuses into intelligently batched sets of tensors that can be directly passed to model
all train, dev and test batches, along with the vocabularies and all hyperparameters for instantiating a corresponding model are stored in serialized dictionary inside <data_path> (default data path is /content/gdrive/My Drive/NMT/data/).
ex) if model is named "my_model", then stored in
/content/gdrive/My Drive/NMT/data/my_model.pkl

step 5 - train model
by default, performs early stopping such that model from best epoch (as measured by BLEU score on the dev set) is stored in NMT/checkpoints/
the most recent checkpoint is always stored, so that can quit training, and continue from where left off at a later date.

step 6 - evaluate model
uses sacreBLEU to measure BLEU score of the model's translations for the test set.