#!/bin/bash
# arg1 - num_merge_ops
# arg2 - vocab_threshold
# arg3 - path to corpuses to be segmented
# arg4 - path denoting where to put segmented corpuses, bpe codes, and vocab files

# learn bpe
echo "learning joint bpe and vocab using $1 merge operations..."
subword-nmt learn-joint-bpe-and-vocab --input "$3"word_train.de "$3"word_train.en -s "$1" -o "$4"bpe_codes --write-vocabulary "$4"vocab.de "$4"vocab.en

# apply bpe to train
echo "applying bpe with vocab threshold of $2 to train..."
subword-nmt apply-bpe -c "$4"bpe_codes --vocabulary "$4"vocab.de --vocabulary-threshold "$2" < "$3"word_train.de > "$4"subword_joint_train.de
subword-nmt apply-bpe -c "$4"bpe_codes --vocabulary "$4"vocab.en --vocabulary-threshold "$2" < "$3"word_train.en > "$4"subword_joint_train.en

# apply bpe to dev and test
echo "applying bpe with vocab threshold of $2 to dev and test..."
subword-nmt apply-bpe -c "$4"bpe_codes --vocabulary "$4"vocab.de --vocabulary-threshold "$2" < "$3"word_dev.de > "$4"subword_joint_dev.de
#subword-nmt apply-bpe -c "$4"bpe_codes --vocabulary "$4"vocab.en --vocabulary-threshold "$2" < "$3"word_dev.en > "$4"subword_joint_dev.en
subword-nmt apply-bpe -c "$4"bpe_codes --vocabulary "$4"vocab.de --vocabulary-threshold "$2" < "$3"word_test.de > "$4"subword_joint_test.de

echo "done"