Encoder-decoder LSTM neural network that translates German into English, implemented in PyTorch.

coming soon:

-support for transformer architecture

-support for convolutional encoder

-support for more language pairs and translation directions




**Optimally efficient:**
* Model forward passes are fully vectorized to operate on batches of variable-length sequences. Intelligent batching is performed so that the number of pad tokens per batch is minimized.
* Beam search inference translates batches of German sentences in parallel, for each of which it computes several candidate translations in parallel.
* Training uses PackedSequence objects to minimize computation inside lstm layer(s).


**Compact yet performant:**
* Achieves a BLEU score of 27 on the IWSLT 2014 test set, even though consists of only x total parameters (makes extensive use of parameter sharing, and preprocesses datasets such that vocabulary size is minimized. see below).
* When use a batch size of 64, this requires ~ y GB of memory to train (this easily fits on, e.g., a Tesla T4 GPU, which is freely available on Google Colab).


**Robust and convenient:**
* Training loop performs early-stopping, and saves both "most-recent-epoch" (so can resume training at later date, or gracefully recover from Colab runtime disconnections, etc.) and "best-so-far" (so can load for later predicting test set, etc.) model checkpoints in Google Drive.
* To facilitate hyperparameter search, automated logging system organizes model checkpoints inside a directory hierarchy, and stores them along with files listing their hyperparameter settings, per-epoch training stats (including dev set bleu scores), and per-epoch greedy predictions for the dev set (to observe translation quality)


**Highly flexible and easily configurable:**
* Specify hyperparameter settings, as well as vocabulary and architectural variations through set of configuration files (see options below). config-importer ensures combination of settings is valid (raises assertion error).
* Supports a variety of architectural, training and inference features, including:
  * Scaled and standard dot product attention mechanisms.
  * Beam search and greedy search inference algorithms.
  * Extraction and application of either word-level or subword-level model vocabularies. Word-level vocab can be constructed by either top_k or thresholding algorithms.














**How To Run:**

Notebooks use Google Drive interface when run in Google Colab environment, so that they can store and load model checkpoints, write training epoch stats, and write translations to files. Place cloned NMT folder inside 'My Drive' folder of Google Drive.

* playground.ipynb

  Open notebook in Google Colab (http://colab.research.google.com) and follow its instructions to:
  * a) interactively input German sentences to observe translation output.
  * b) replicate reported BLEU score on the IWSLT 2014 test set.





* NMT-driver.ipynb

  follow notebook instructions to:
  * a) replicate the 6 total steps -- each in its own cell -- needed to preprocess the datasets, as well as train and evaluate a model.
  * b) run unit tests to show correctness of model implementations.

  (all preprocessing steps save their outputs to corresponding files, so that if wish to change a given step, do not need to re-run prior steps).

  step 1 - Apply series of pre-trained Stanza (Stanford CoreNLP) processors, which perform tokenization, multi-word token expansion, and part-of-speech tagging
  * processor outputs are saved to pickle files with the prefix "stanza_" inside corpuses/iwslt16_en_de/stanza_outputs/ (each corpus is processed and saved in pieces, bc entire output does not fit in memory)


  step 2 - Truecase the corpuses using linguistic heuristics that leverage morphological data produced in step 1
  * truecased corpuses are saved to files with the prefix "word_" inside corpuses/iwslt16_en_de/truecased/
    * This step acts to reduce the vocabulary (and therefore embeddings matrix sizes) of the model, since a word no longer appears in capitalized and lowercase variants (which would give it two slots in the embeddings table) for exclusively syntactic purposes, e.g., starting a sentence, quotation, etc. (To produce fully cased predictions, casing is heuristically applied during post-processing).
    * Further, this allows the model to consolidate a word's meaning into a single entry, facilitating training.
    * coming soon: additional preprocessing steps, such as compound-splitting and punctuation-collapsing.


  step 3 - (optional) Segment corpuses of words into corpuses of subwords (using byte-pair-encodings of https://github.com/rsennrich/subword-nmt)
  * segmented corpuses are saved to files with the prefix "subword_joint_" or "subword_ind_" inside corpuses/iwslt16_en_de/subword_segmented/, depending on if learn a joint vocabulary or separate, independent vocabularies, respectively, for the source and target languages.


  step 4 - Convert corpuses into batched sets of tensors that can be directly passed to model
  * all train and dev batches (and corresponding data, e.g., attention masks, indices for unsorting the source sentences, etc.), vocabularies, and hyperparameters for instantiating a corresponding model are stored in model_data.pkl inside checkpoints/\<model_name\>


  step 5 - Train model
  * model from best epoch (as measured by BLEU score on the dev set) and most recent model checkpoints are stored in checkpoints/\<model_name\>


  step 6 - Evaluate model
  * uses sacreBLEU to measure BLEU score of the model's translations for the test set.











**Default architecture overview:**
Overview of the pre-trained model (stored as pretrained inside X).
(note: nearly all of these layers can be altered by setting the configuration files (see below))
To reduce the total number of parameters, the following weights are shared:
* Encoder and decoder share the same embeddings matrix (i.e., model uses a joint German-English subword vocabulary).
* Decoder's output matrix weights are tied to the embeddings table.

Encoder:
* A single bidirectional lstm layer reads the source sentence in the forward and reverse directions. For a given source word, its forward and backward hidden representations are concatenated together. The set of concatenated representations are then passed through a custom dropout layer (where the same dropout mask is applied to each timestep).
* An additional layer then projects these states into a set of "keys" (for use by the decoder's attention mechanism) with the same dimensionality as the decoder's hidden layer size, followed by a tanh nonlinearity. A different set of weights is used to convert the hidden state from the final encoder timestep into the initial decoder lstm hidden state.

Decoder:
* A single lstm layer reads the target sentence in the forward direction, and custom dropout is then applied as in the encoder.
* The attention mechanism is then applied, where for each decoder hidden state (the "query"), the dot product is computed with each of the keys, which are then normalized into a set of weights for computing a weighted average of each encoder hidden state, called the "context".
* This context vector is then concatenated to the decoder hidden state, and passed through an "attention layer", that projects it to the same dimensionality as the embeddings, followed by a tanh nonlinearity. This is then projected by the embeddings matrix to a |V|-dimensional vector, where |V| is the vocab size, and entry i holds the score that the i'th subword of the shared vocabulary is the next word of the target sentence.
* During training, the log softmax is applied, and the cross-entropy loss is computed with respect to the actual next target word. 
* During (greedy) inference, the target word with the highest score is passed as the input for the next decoder timestep.



**Corpus Overview**
Model is trained on the IWSLT 2016 corpus, which consists of ~200,000 German-English pairs of text spans, where the English span is the ground-truth translation of the German span. A text span is usually one (but sometimes multiple) sentence(s). This corpus is comprised of transcribed TED talks, which are "challenging due to their variety in topics, but are very benign as they are very thoroughly rehearsed and planned, leading to easy to recognize and translate language." 





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

