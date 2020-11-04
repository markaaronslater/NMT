Encoder-decoder LSTM neural network that translates German into English, implemented in PyTorch.

coming soon:

-support for transformer architecture

-support for more language pairs and translation directions




**Optimally efficient:**
* no unnecessary loops in training and inference code.
* implements intelligent batching so that number of pad tokens in batches (which contain variable-length sequences) is minimized for both training and inference.
* training uses PackedSequence objects to avoid needless computation inside lstm layer(s).

**Compact yet performant:**
* employs variety of x in order to obtain BLEU score of 29 on y, even though uses only z total parameters (requires roughly Y GB of memory to train on Colab GPU, when use train and dev bsz's of 64), by using advanced techniques like weight sharing.
easily fits in gpu memory of free cloud providers, like Google Colab
requires only modestly sized 
default encoder and decoder each only contain single lstm layer.
encoder lstm is bidirectional, and its states are projected
evaluated using sacrebleu, the most strict and reliable bleu score calculator.



**Supports variety of model, training and prediction features, including:**
* scaled and standard dot product attention mechanisms.
* beam search and greedy search inference algorithms.
* extraction and application of either word-level or subword-level model vocabularies.

**Highly flexible and easily configurable:**
* specify hyperparameter settings and architectural variations through set of configuration files (see options below).
* config-importer ensures combination of settings is valid (raises assertion error).

**Robust and convenient:**
* training algorithm performs early-stopping and saves both "most-recent" (so can resume training at later date, or gracefully recover from Colab runtime disconnections, etc.) and "best-so-far" (so can extract for later predicting test set, etc.) model checkpoints in Google Drive.
* automatically logs hyperparameter settings used by given model inside its checkpoints folder, along with training stats and dev set bleu scores, to facilitate hyperparameter search.
* train model from scratch or resume training a checkpoint where left off
* automates organization of models within file system 


**Corpus preprocessing**

all preprocessing steps save their outputs to corresponding files (see How to Run, below), so that if wish to change a given step, do not need to re-run prior steps.
* all preprocessing performed starting from parallel sets of German and English training corpuses, "train.de" and "train.en", where line i of "train.en" is the ground-truth English translation of line i of "train.de".
* Calls a series of pre-trained Stanford CoreNLP (Stanza) processors that perform tokenization, multi-word token expansion, and part-of-speech tagging on each corpus.
* This part-of-speech information is then leveraged by truecasing algorithm that employs linguistic heuristics to decide whether or not to convert a word to lowercase (so that a word's meaning is not distributed across capitalized and lower-case embeddings, size of vocabulary is reduced, etc.).
* These truecased corpuses of words are then optionally segmented into corpuses of subwords (employing byte-pair-encodings of https://github.com/rsennrich/subword-nmt-nmt).
* Lastly, vocabularies are built and applied to the corpuses, from which intelligently batched sets of tensors are constructed ahead of training (to save time in forward pass), and are packaged along with all other relevant data (e.g., masks used by decoder during attention computation).
* coming soon: additional preprocessing steps, such as compound-splitting and punctuation-collapsing.














**How To Run:**

notebooks use Google Drive interface when run in Google Colab environment, so that they can store and load model checkpoints, write training epoch stats, and write translations to files. Place cloned NMT folder inside 'My Drive' folder of Google Drive.

* playground.ipynb

  open notebook in Google Colab (http://colab.research.google.com) and follow its instructions to:
  * observe translations of sample German sentences into English.
  * interactively input German sentences to observe translation output, and coming soon: compare translation side-by-side with Google translate's output.





* NMT-driver.ipynb

  follow notebook instructions to:
  * a) replicate any subset of the 6 steps (see below), each in its own cell, needed to preprocess the data, train, and evaluate a model,
  * b) run unit tests to show correctness of model implementations, or
  * c) load a pre-trained model checkpoint to determine BLEU score on test set.


  step 0 - make necessary folders not included in repo, e.g., checkpoints, stanza_outputs, truecased, subword_segmented
  
  step 1 - apply Stanza processors, which perform tokenization, multi-word token expansion, and part-of-speech tagging
  * processor outputs are saved to pickle files with the prefix "stanza_" inside corpuses/iwslt16_en_de/stanza_outputs/
  * each corpus is processed in pieces, bc entire output does not fit in memory.


  step 2 - truecase the corpuses using linguistic heuristics that leverage morphological data produced by morphological data tagger
  * truecased corpuses are saved to files with the prefix "word_" inside corpuses/iwslt16_en_de/truecased/ (they are used directly by models that employ a word-level vocabulary).

  step 3 - segment corpuses of words into corpuses of subwords
  * segmented corpuses are saved to files with the prefix "subword_joint_" or "subword_ind_" inside corpuses/iwslt16_en_de/subword_segmented/, depending on if learn a joint vocabulary or separate, independent vocabularies, respectively, for the source and target languages.

  step 4 - convert corpuses into batched sets of tensors that can be directly passed to model
  * all train, dev and test batches, along with the vocabularies and all hyperparameters for instantiating a corresponding model are stored in serialized dictionary inside data/


  step 5 - train model
  * model from best epoch (as measured by BLEU score on the dev set) and most recent model checkpoints are stored in checkpoints/


  step 6 - evaluate model
  * uses sacreBLEU to measure BLEU score of the model's translations for the test set.






**How to use configuration files:** 

Encoder Options | Default Setting | Possible Settings | Comments
------| --------- | --------- | --------------------------------------
bidirectional | True | boolean | if true, use bidirectional instead of unidirectional encoder
project | True | boolean | whether to project the concatenated forward and backward encoder states back to original dimensionality (so that same size as decoder hidden states when computing attention) (only used if bidirectional == True)
reverse_src | False | boolean |  whether to pass src sentence to encoder in reverse order (only used if bidirectional == False)
decoder_init_scheme | layer_to_layer | [layer_to_layer, final_to_first] | how the hidden states from the final timestep of the encoder are used to construct the initial hidden state of the decoder. can have each layer of encoder directly initialize corresponding layer of decoder (requires same number of layers in each) or have final layer of encoder initialize first layer of decoder (with all non-first decoder layers initialized as zeros)
enc_input_size | 300 | int | embedding size
enc_hidden_size | 600 | int |
enc_num_layers | 1 | int | 
enc_lstm_dropout | 0.0 | float | applied to all non-final lstm layers. different dropout mask applied for each timestep (only used if enc_num_layers > 1)
enc_dropout | 0.2 | float | applied after final lstm layer. same dropout mask applied for each timestep



Decoder Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
tie_weights | False | boolean | whether or not to share weights in the embeddings matrix and the output matrix (requires input and hidden sizes to have same dimensionality)
attention_fn | dot_product | [dot_product, scaled_dot_product, none] | whether or not to include attention mechanism, and if so, which one
attention_layer | False | boolean | whether or not to project the attentional representations (which are contexts concatenated to hidden states) from 2*hidden_size back to hidden_size (along with a nonlinearity) before projecting them to vocab_size in output layer. only used if attention_fn is not "none". must be set to True if tie_weights == True, bc in this case hidden_size == input_size
beam_width | 10 | int | how many candidate translations to consider the running probabilities of (for a single src sentence) at any given moment during beam search inference
decode_slack | 20 | int | max number of words to predict beyond the length of corresponding source sentence length, before "times out" (controls strength of heuristic that an English translation will be a similar number of words as German source sentence) (cuts off translations that never produce \<end-of-sentence\> symbol)
dec_input_size | 300 | int | embedding size
dec_hidden_size | 600 | int | 
dec_num_layers | 1 | int |  
dec_lstm_dropout | 0.0 | float | applied to all non-final lstm layers. different dropout mask applied for each timestep (only used if dec_num_layers > 1)
dec_dropout | 0.2 | float | applied after final lstm layer. same dropout mask applied for each timestep



Training Options | Default Setting | Possible Settings | Comments
--------| --------------- | ----------------- | --------
early_stopping | True | boolean | whether or not to quit training early if bleu scores on dev set do not improve for <early_stopping_threshold> epochs in a row (see below)
early_stopping_threshold | 5 | int | 
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