
does not make a ton of sense to make one end-to-end preprocessing function, bc the stanford nlp stanza pos-tagger, tokenizer, segmenter step is time-consuming, so makes more sense to do that a single time & save the results to files.
next, use the processor outputs to normalize the corpuses.
finally, convert the normalized corpuses into batches of tensors that can directly pass to the model.

for practical reasons, preprocessing is split into 2 distinct phases.

2 distinct phases of corpus preprocessing.
then, single phase of converting phase-2 preprocessed corpuses into model inputs.

-> after each step, saves outputs to text and/or pickle files so never need repeat a step


(1) convert corpuses into model inputs:

(1.0) make sure the following are installed:
        stanza (pip install stanza)
        subword-nmt (pip install subword-nmt)
        ??sacrebleu??

(1.1) run preprocess_phase1() fn
-> run on commandline as preprocess_phase1.py (use argparse)

-performs phase 1 of preprocessing (writes outputs to pickle files)
   -sentence segmentation
   -sentence tokenization
   -produces morphological data, some of which is needed for phase 2: e.g.,
      -part-of-speech tags
      -number
      -gender


(1.2) run preprocess_phase2() fn
run on commandline as preprocess_phase2.py (use argparse)

-performs phase 2 of preprocessing
   -concatenates chunks of outputs of stanfordnlpprocessor into total corpuses
   -intelligent decasing (uses morphological data from phase 1)
   -writes decased corpuses to files


optional - only necessary if using subword vocab (rather than word vocab)
   (1.3) run jointBPE.sh or BPE.sh
      -performs word-splitting on the decased files
      -builds subword vocab
         -jointBPE.sh learns a joint vocab for both src and trg. BPE.sh learns separate vocabs for src and trg


(1.4) run build_model_inputs() fn
run on commandline as build_model_inputs.py (use argparse)

-converts phase-2 preprocessed corpuses into model inputs (writes outputs to pickle files)
   -loads tokenized corpuses
   -builds and stores references
   -adds <start-of-sentence> and <end-of-sentence> tags to target sentences
   -builds or loads vocabulary (former, if vocab_type=word)
   -applies vocabulary to corpuses
      -replaces OOV words with <unknown> tokens (if vocab_type=word)
      -replaces words with their indices
   -converts lists of indices to tensors padded with <pad> tokens to be used as encoder inputs, decoder inputs, and decoder targets, as well as builds other data structures needed by model, e.g., masks, etc.



(2) train model:
(2.1) build model using a config file or by passing arguments via argparse
(2.2) train model from scratch or continue training a checkpoint



(3) test model:
(3.1) choose a model checkpoint, perform inference on dev set

(3.2) eval using sacrebleu script
run sacrebleu perl script

