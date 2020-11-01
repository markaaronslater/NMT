import torch
import torch.nn.functional as F


def initialize_beams(dists_1, hidden_1, hp):
    bsz = hp["bsz"]
    b = hp["b"]
    nl = hp["nl"]
    d_hid = hp["d_hid"]

    word_likelihoods = F.log_softmax(dists_1, dim=-1)
    top_word_likelihoods, top_words = torch.topk(word_likelihoods, b) # each is (bsz x b)
    # initialize sequences, which holds running translations.
    sequences = top_words.unsqueeze(2) # (bsz x b x 1)
    # seq_likelihoods holds log probabilities of running translations.
    # each consists of single word, so initialized as word likelihoods.
    seq_likelihoods = top_word_likelihoods # (bsz x b)
    
    h_1, c_1 = hidden_1
    h_1 = h_1.unsqueeze(2).expand(-1, -1, b, -1).contiguous() # (nl x bsz x b x d_hid)
    c_1 = c_1.unsqueeze(2).expand(-1, -1, b, -1).contiguous()
    # reshape so can pass thru lstm (within a given beam of batch, 
    # each of its sequences was produced by same hidden state).
    h_i = h_1.view(nl, bsz*b, d_hid)
    c_i = c_1.view(nl, bsz*b, d_hid)

    return sequences, seq_likelihoods, (h_i, c_i), top_words


def expand_beams(dists_i, seq_likelihoods, hp):
    bsz = hp["bsz"]
    b = hp["b"]
    nl = hp["nl"]
    d_hid = hp["d_hid"]
    v = hp["v"]

    word_likelihoods = F.log_softmax(dists_i.view(bsz, b, v), dim=-1) # (bsz x b x v) 
    top_word_likelihoods, top_words = torch.topk(word_likelihoods, b) # -> each is (bsz x b x b)
    top_words = top_words.view(bsz, -1) # (bsz x b*b)
    candidate_likelihoods = seq_likelihoods.unsqueeze(2) + top_word_likelihoods # (bsz x b x b)
    candidate_likelihoods = candidate_likelihoods.view(bsz, -1) # (bsz x b*b)

    return candidate_likelihoods, top_words


# select the best sequences of the beams, orient them with the hidden states
# that produced them, and prepare corresponding inputs for next decode timestep.
def update_beams(sequences, next_words, top_candidates, hidden_i, timestep, hp):
    bsz = hp["bsz"]
    b = hp["b"]
    nl = hp["nl"]
    d_hid = hp["d_hid"]
    h_i, c_i = hidden_i

    # update sequences
    seq_indices = (top_candidates // b).unsqueeze(2).expand(-1,-1,timestep-1).contiguous() # (bsz x b x (i-1))
    sequences = torch.cat((torch.gather(sequences, 1, seq_indices), next_words.unsqueeze(2)), dim=2) # (bsz x b x i), where i is num decoding time steps

    # update hidden states
    # separate beams
    h_i = h_i.view(nl, bsz, b, d_hid)
    c_i = c_i.view(nl, bsz, b, d_hid)
    # extract hidden states that produced this round of sequence candidates,
    # so that can continue exploring them in next timestep.
    hidden_indices = (top_candidates // b).view(1, bsz, b, 1).expand(nl, -1, -1, d_hid).contiguous()
    h_i = torch.gather(h_i, 2, hidden_indices)
    c_i = torch.gather(c_i, 2, hidden_indices)
    # reshape so can pass back into lstm
    h_i = h_i.view(nl, bsz*b, d_hid)
    c_i = c_i.view(nl, bsz*b, d_hid)
    
    return sequences, (h_i, c_i)


def write_finished_translations(translation, sequences, finished, eos, timestep):
    # termination condition: most probable seq produced by beam ends in eos.
    predicted_eos = sequences[:, 0, -1] == eos
    # -> (bsz, ) booltensor containing a 1 for the sequences whose beams
    # just produced sequence ending in eos as their most likely sequence.
    just_finished = (finished.logical_not()).logical_and(predicted_eos)
    # entry j is True if seq j finished being translated this timestep.
    # obtain indices to extract the sequences that just finished.
    just_finished_indices = torch.nonzero(just_finished, as_tuple=False).squeeze(1) # (num_just_finished, )
    # (this loop and the exit loop together run a total of <bsz> iterations,
    # so implementation is still fully vectorized).
    for j in just_finished_indices:
        translation[j,:timestep] = sequences[j][0]

    finished.logical_or_(predicted_eos) # update finished