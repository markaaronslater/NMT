import pytest # so can use pytest.raises() method

from encoderdecoder import x



# neural network correctness tests: unregularized model of sufficient capacity can overfit first 10 sentences of training set.

# 1-achieve ~zero loss on first 10 sentences of training set
trainBatches = getBatches(trainingPairs[:10], 10, device)
devBatches = getDevBatches(corpuses["train.de"][:10], 10, device)

# 2-perfectly predict the first 10 training sentences (BLEU == 1)
def my_abs(x):
    if x < 0:
        return -1 * x
    else:
        return x

# neural network sanity checks
# 1-tensors are of correct shapes


# 2-initial loss (when initialize params with small, near-zero values) is approximately log C, where C is the vocabulary size

# 3-increasing L2-regularization strength increases the loss
def test_my_abs():
    assert my_abs(2) == 2
    assert my_abs(-2) == 2

    with pytest.raises(TypeError):
       my_abs("hello")

def test_always_fails():
    assert True == False