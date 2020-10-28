import pytest # so can use pytest.raises() method

from encoderdecoder import x



# neural network correctness tests: unregularized model of sufficient capacity can overfit first 10 sentences of training set.
###!!!remember to set decode slack very high

# combos to test:
# unidirectional encoder, no attention
# bidirectional encoder, no attention
# unidirectional encoder, attention
# bidirectional encoder, attention

# layer_to_layer init_scheme
# final_to_first init_scheme

# dot_product_attn
# scaled_dot_product_attn

# apply attention_layer that projects attentional representations back to original hidden_size before projecting to vocab size
# directly project attentional representations to vocab size

# tie-weights = true
# tie-wieghts = false

# word vocab thresholded
# word vocab top_k
# subword vocab independent
# subword vocab joint
# subword vocab pos-tag-concatenated

# beam search
# greedy search

# early stopping
# storing and loading

# 1-achieve ~zero loss on first 10 sentences of training set
# ??should i use sum instead of mean loss, so can more easily observe whether loss is decreasing??
trainBatches = getBatches(trainingPairs[:10], 10, device)
devBatches = getDevBatches(corpuses["train.de"][:10], 10, device)

# 2-perfectly predict the first 10 training sentences (BLEU == 1)
def my_abs(x):
    if x < 0:
        return -1 * x
    else:
        return x

# neural network sanity checks
# 1-outputted tensors are of correct shapes


# 2-initial loss (when initialize params with small, near-zero values) is approximately log C, where C is the vocabulary size

# 3-increasing L2-regularization strength increases the loss
def test_my_abs():
    assert my_abs(2) == 2
    assert my_abs(-2) == 2

    with pytest.raises(TypeError):
       my_abs("hello")

def test_always_fails():
    assert True == False