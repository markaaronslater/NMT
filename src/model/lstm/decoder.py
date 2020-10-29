import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from beam_search_utils import initialize_beams, expand_beams, update_beams, write_finished_translations

class Decoder(nn.Module):
    def __init__(self, hyperparams):
        super(Decoder, self).__init__()
        self.sos_idx = hyperparams["sos_idx"]
        self.eos_idx = hyperparams["eos_idx"]
        
        # hyperparameters settings
        self.vocab_size = hyperparams["trg_vocab_size"]
        self.input_size = hyperparams["dec_input_size"]
        self.hidden_size = hyperparams["dec_hidden_size"]
        self.num_layers = hyperparams["dec_num_layers"]
        self.dropout = hyperparams["dec_dropout"]
        self.project_att_states = hyperparams["attention_layer"]
        self.attention = hyperparams["attention_fn"] != "none"
        self.inference_alg = hyperparams["inference_alg"]
        # # only project hidden state with intermediate layer if use attention mechanism
        # assert (self.attention or not self.project_att_states)

        if self.attention:
            if hyperparams["attention_fn"] == "dot_product":
                self.attend = dot_product_attn
            elif hyperparams["attention_fn"] == "scaled_dot_product":
                self.attend = scaled_dot_product_attn
            else:
                raise NameError(f"specified an unsupported attention mechanism: {hyperparams['attention_fn']}")
            
        if self.inference_alg == "beam_search": 
            self.predict = beam_search_translate
        elif self.inference_alg == "greedy_search": 
            self.predict = greedy_search_translate
        else:
            raise NameError(f"specified an unsupported inference algorithm: {hyperparams['inference_alg']}")

        self.beam_width = hyperparams["beam_width"]
        self.decode_slack = hyperparams["decode_slack"]

        # architecture
        self.embed = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)

        # whether or not we include an additional layer that projects attentional states back to hidden_size influences size of output matrix
        if not self.attention:
            self.out = nn.Linear(self.hidden_size, self.vocab_size)
        elif self.project_att_states:
            # employ an additional layer that projects hidden states back to hidden_size before passing thru output layer.
            self.attention_layer = nn.Linear(2*self.hidden_size, self.hidden_size)
            # also influences size of output matrix.
            self.out = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.out = nn.Linear(2*self.hidden_size, self.vocab_size)

        # use same matrix for embedding target words as for predicting probability distributions over those words
        if bool(hyperparams["tie_weights"]):
            #assert self.hidden_size == self.input_size
            ###???how do these dimensions work out, again???
            self.out.weight = self.embed.weight


    # inference uses sampling: predicts next word given words it previously predicted.
    # -> must predict single token at a time, so does not use PackedSequences.
    # training uses teacher-forcing: predicts next word given ground-truth previous words)
    # -> can pass entire ground-truth trg sentence at once, so uses PackedSequences.
    # (inference and training fwd passes therefore handled separately).
    def forward(self, decoder_inputs, encoder_states, initial_state):
        if not self.train:
            return self.predict(decoder_inputs, encoder_states, initial_state)
        else:
            embs = self.embed(decoder_inputs["in"]) # (bsz x max_trg_len x input_size)
            packed_input = pack_padded_sequence(embs, decoder_inputs["lengths"], batch_first=True)
            packed_output, _ = self.lstm(packed_input, initial_state) # (total_len x hidden_size)
            # (where total_len is the total number of non-pad tokens in the batch).
            if self.attention:
                decoder_states, _ = pad_packed_sequence(packed_output, batch_first=True) # (bsz x max_trg_len x hidden_size)
                attentional_states = self.attend(decoder_states, encoder_states, decoder_inputs["mask"]) # (bsz x max_trg_len x 2*hidden_size)
                attentional_states = pack_padded_sequence(attentional_states, decoder_inputs["lengths"], batch_first=True).data # (total_len x 2*hidden_size)
            else:
                attentional_states = packed_output.data

            del packed_input, packed_output
            if self.project_att_states:
                attentional_states = F.tanh(self.attention_layer(attentional_states)) # (total_len x hidden_size)
            
            dists = self.out(attentional_states) # (total_len x vocab_size)
            # for each sequence of batch, for each time step, holds predicted logits over next words.

            return dists


    # -returns concatenation of the queries with their context vectors (bsz x (2*hidden_size)).
    # -queries (decoder states) is (bsz x max_trg_len x decoder_hidden_size).
    # -keys (encoder states) is (bsz x max_src_len x encoder_hidden_size).
    # -mask is (bsz x 1 x max_src_len).
    # mask[i][0][k] = 1 wherever keys[i][k] holds hidden state corresponding
    # to a pad token (i.e., does not correspond to actual src token
    # so dn include it in the weighted average).
    # -each of the max_trg_len decoder states in queries
    # computes attention over same set of encoder states, so dim 1 of mask
    # broadcasts over each query).
    def dot_product_attn(self, queries, keys, mask):
        # entry i,j,k holds dot product of query j (hidden state of decoder timestep j)
        # with key k (hidden state of encoder timestep k), for i'th sequence of batch.
        scores = torch.bmm(queries, keys.transpose(1,2)) # (bsz x max_trg_len x max_src_len)
        # everywhere mask holds a 1, fill the corresponding entry of scores with negative infinity.
        # -> this overwrites the dot products with encoder states of pad tokens, 
        # so when apply softmax, their weight gets set to zero.
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores, dim=2)
        # context vector for a given query is an average over all encoder states, weighted by their dot product with the query.
        contexts = torch.bmm(weights, keys) # (bsz x max_trg_len x hidden_size)
        del scores, weights

        # concatenate contexts with the original decoder states
        attentional_states = torch.cat((contexts, queries), dim=2) # (bsz x max_trg_len x 2*hidden_size)

        return attentional_states


    def scaled_dot_product_attn():
        pass


    # applied to multiple sequences in parallel, for each of which it predicts entire beam in parallel.
    # each beam's sequences maintained in descending order by likelihood.
    def beam_search_translate(self, decoder_inputs, encoder_states, initial_state):
        bsz = encoder_states.size(0) # batch size
        b = self.beam_width
        # hyperparameters to be referenced in util functions
        hp = {  "bsz":bsz,
                "b":b,
                "nl":self.num_layers,
                "d_hid":self.hidden_size,
                "v":self.vocab_size
             }
        eos = torch.tensor([self.eos_idx], dtype=torch.long)
        T = decoder_inputs["max_src_len"] + self.decode_slack # max number of decoder time steps
        # whenever a beam's most probable sequence produces eos for the first time, copy it to corresponding row of translation
        translation = torch.zeros((bsz, T), dtype=torch.long)
        # stop decoding when the most likely sequence of each beam has predicted the eos token.
        finished = torch.zeros((bsz,), dtype=torch.bool) # entry j is 1 if have finished translating sentence j

        # handle timestep 1 separately from the rest, bc initializes the beams 
        # (employs a beam_width of 1, rather than b), each of which was produced
        # by its src sequence's final hidden state.
        sos_tokens = torch.full((bsz, 1), self.sos_idx, dtype=torch.long)
        input_0 = self.embed(sos_tokens) # (bsz x 1 x input_size)
        h_0, c_0 = initial_state
        dists_1, (h_1, c_1) = self.decode_step(input_0, (h_0, c_0), decoder_inputs["mask"], encoder_states) 
        # -> dists_1 is (bsz x v), h_1 and c_1 are (nl x bsz x hidden_size)
        sequences, seq_likelihoods, (h_i, c_i), top_words = initialize_beams(dists_1, (h_1, c_1), hp)
        input_i = self.embed(top_words.view(bsz*b, 1)) # (bsz*b x 1 x input_size)

        ### edge case: immediately predicts eos (only plausible in early stages of overfitting during unit tests)
        write_finished_translations(translation, sequences, finished, eos, 1)
        if finished.all():
            return translation
        #########################

        # timesteps 2 and onward handled inside loop.
        # iter i takes sequences of length i-1 and extends them to length i.
        for i in range(2, T):
            # the encoder states for a given source sequence in the batch must
            # be repeated b times, so that can compute attention with
            # hidden state for each target sequence in corresponding beam.
            dists_i, (h_i, c_i) = self.decode_step(input_i, (h_i, c_i), decoder_inputs["mask"], encoder_states.repeat(1,b,1)) # (bsz*b x v)
            # each of the b sequences of a beam generates b successors, providing b*b candidates for the next beam.
            candidate_likelihoods, top_words = expand_beams(dists_i, seq_likelihoods, hp)
            # identify the top b most likely of the b*b candidates.
            top_candidate_likelihoods, top_candidates = torch.topk(candidate_likelihoods, b) # (bsz x b)
            seq_likelihoods = top_candidate_likelihoods # update seq likelihoods
            next_words = torch.gather(top_words, 1, top_candidates) # words appended to beam sequences
            # fill beams in descending order with most likely sequences,
            # along with the hidden states that produced them.
            sequences, (h_i, c_i) = update_beams(sequences, next_words, top_candidates, i, hp)
            # must write a translation as soon as it's finished, bc if
            # continue extending it, resulting sequence will be of extremely
            # low probability (words never follow <eos>) and therefore lost.
            write_finished_translations(translation, sequences, finished, eos, i)
            if finished.all():
                return translation

            input_i = self.embed(next_words.view(bsz*b, 1)) # (bsz*b x 1 x input_size)

        # copy the translations whose beams never produced a most-likely sequence ending in eos.
        never_finished_indices = torch.nonzero(finished.logical_not()).squeeze()# (num_never_finished, )
        for j in never_finished_indices:
            translation[j,:i] = sequences[j][0]
            
        # translation now contains sequences with eos symbols. will extract the portion prior to the eos symbol inside postprocess.py
        return translation

    
    # compute the decoder states one at a time, using prediction from previous time step as input to current timestep.
    # longest src sentence of dev batch used as heuristic for deciding max num time steps for translating target words.
    # (decode_slack is used to cut off translations that might otherwise never end).
    # returns (bsz x T) tensor, where T is number of decode time steps (at least 1, and at most max_src_len + decode_slack), holding the batch of predicted translations.
    def greedy_search_translate(self, decoder_inputs, encoder_states, initial_state):
        bsz = encoder_states.size(0)
        eos = torch.tensor([self.eos_idx], dtype=torch.long)
        T = decoder_inputs["max_src_len"] + self.decode_slack # max number of decoder time steps

        decoder_in = torch.full((bsz, 1), self.sos_idx, dtype=torch.long) # (bsz x 1)
        # initialize running translation (each decode step concatenates to it)
        translation = torch.tensor([], dtype=torch.long)
        # stop decoding when every sentence has predicted the eos token.
        finished = torch.zeros((bsz,), dtype=torch.bool)
        # -> entry j is 1 if have finished translating sentence j

        input_i = self.embed(decoder_in) # (bsz x 1 x input_size)
        (h_i, c_i) = initial_state
        # iter i takes sequences of length i and extends them to length i+1.
        for i in range(T): 
            dists_i, (h_i, c_i) = self.decode_step(input_i, (h_i, c_i), decoder_inputs["mask"], encoder_states)
            # greedy: take argmax to get position of each dist containing highest score
            preds_i = torch.argmax(dists_i, 1).unsqueeze(1) # (bsz x 1)
            # concat to the running translations for each seq of the batch
            translation = torch.cat((translation, preds_i), dim=1) # (bsz x (i+1))
            finished = finished.logical_or(preds_i == eos)
            if finished.all():
                break

            # predictions of this time step serve as inputs for next time step
            input_i = self.embed(preds_i) # (bsz x 1 x input_size)

        return translation


    # modularize the shared computation of greedy and beam search decoding.
    # fwd pass for batch of length-1 sentences, corresponding to time step i.
    def decode_step(self, input_i, hidden, mask, encoder_states):
        output_i, (h_i, c_i) = self.lstm(input_i, hidden) # output_i is (bsz x 1 x hidden_size)
        if self.attention:
            attentional_states_i = self.attend(output_i, encoder_states, mask).squeeze()
            # -> (bsz x 2*hidden_size)
        else:
            attentional_states_i = output_i.squeeze() # (bsz x hidden_size)
        if self.project_att_states:
            attentional_states_i = F.tanh(self.attention_layer(attentional_states_i)) 

        dists_i = self.out(attentional_states_i) # (bsz x vocab_size)

        return dists_i, (h_i, c_i)