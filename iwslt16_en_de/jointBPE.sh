#!/bin/bash
# arg1 - numMerges
# arg2 - vocabThreshold
# arg3 - path to normalized corpuses

# learn bpe
echo "learning joint bpe and vocab using $1 merge operations..."
subword-nmt learn-joint-bpe-and-vocab --input "$3"/decased_train.de "$3"/decased_train.en -s "$1" -o "$3"/bpe_codes --write-vocabulary "$3"/vocab.de "$3"/vocab.en

# apply bpe to train
echo "applying bpe with vocab threshold of $2 to train..."
subword-nmt apply-bpe -c "$3"/bpe_codes --vocabulary "$3"/vocab.de --vocabulary-threshold "$2" < "$3"/decased_train.de > "$3"/train.BPE.de
subword-nmt apply-bpe -c "$3"/bpe_codes --vocabulary "$3"/vocab.en --vocabulary-threshold "$2" < "$3"/decased_train.en > "$3"/train.BPE.en

# apply bpe to dev and test
echo "applying bpe with vocab threshold of $2 to dev and test..."
subword-nmt apply-bpe -c "$3"/bpe_codes --vocabulary "$3"/vocab.de --vocabulary-threshold "$2" < "$3"/decased_dev.de > "$3"/dev.BPE.de
subword-nmt apply-bpe -c "$3"/bpe_codes --vocabulary "$3"/vocab.en --vocabulary-threshold "$2" < "$3"/decased_dev.en > "$3"/dev.BPE.en

subword-nmt apply-bpe -c "$3"/bpe_codes --vocabulary "$3"/vocab.de --vocabulary-threshold "$2" < "$3"/decased_test.de > "$3"/test.BPE.de

echo "done"