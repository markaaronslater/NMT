import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from math import sqrt

from src.model.lstm.beam_search_utils import initialize_beams, expand_beams, update_beams, write_finished_translations

class Decoder(nn.Module):
    def __init__(self, hyperparams, inference_alg="greedy_search"):
        super(Decoder, self).__init__()
        self.sos_idx = hyperparams["sos_idx"]
        self.eos_idx = hyperparams["eos_idx"]
        
        # hyperparameters settings
        self.vocab_size = hyperparams["trg_vocab_size"]
        self.input_size = hyperparams["dec_input_size"]
        self.hidden_size = hyperparams["dec_hidden_size"]
        self.num_layers = hyperparams["dec_num_layers"]
        self.lstm_dropout = hyperparams["dec_lstm_dropout"]
        self.project_att_states = hyperparams["attention_layer"]
        self.attention = hyperparams["attention_fn"] != "none"
        self.tie_weights = hyperparams["tie_weights"]
        self.add_drop_layer = hyperparams["dec_dropout"] > 0
        self.out_drop = hyperparams["dec_dropout"]
        self.device = hyperparams["device"]

        if self.attention:
            if hyperparams["attention_fn"] == "dot_product":
                self.scaled = False
            elif hyperparams["attention_fn"] == "scaled_dot_product":
                self.scaled = True
                self.sqrt_hs = sqrt(self.hidden_size)
            else:
                raise NameError(f"specified an unsupported attention mechanism: {hyperparams['attention_fn']}")
        
        self.set_inference_alg(inference_alg)
        self.beam_width = hyperparams["beam_width"]
        self.decode_slack = hyperparams["decode_slack"]

        # architecture
        self.embed = nn.Embedding(self.vocab_size, self.input_size)
        ### added ###
        #self.embed.weight.data.uniform_(-.1, .1)
        ##############

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.lstm_dropout, batch_first=True)

        # whether or not we include an additional layer that projects
        # attentional states back to hidden_size influences size of output matrix.
        if not self.attention:
            self.out = nn.Linear(self.hidden_size, self.vocab_size)
        elif self.project_att_states:
            # adds additional layer between attention mechanism and output.
            if self.tie_weights:
                # use same matrix for embedding target words as for
                # predicting probability distributions over those words.
                self.attention_layer = nn.Linear(2*self.hidden_size, self.input_size)
                # also influences size of output matrix.
                self.out = nn.Linear(self.input_size, self.vocab_size)
            else:
                self.attention_layer = nn.Linear(2*self.hidden_size, self.hidden_size)
                self.out = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            self.out = nn.Linear(2*self.hidden_size, self.vocab_size)
        
        if self.tie_weights:
            # embed and out are each (vocab_size x input_size).
            #   -> in fwd pass, out's transpose is multiplied on the right:
            #      h * out^T + bias
            self.out.weight = self.embed.weight

        #### added #####
        #self.out.bias.data.zero_()
        ################
        # if self.add_drop_layer:
        #     self.dropout_layer = nn.Dropout(p=hyperparams["dec_dropout"])
        
            

    # inference uses sampling: predicts next word given words it previously predicted.
    # -> must predict single token at a time, so does not use PackedSequences.
    # training uses teacher-forcing: predicts next word given ground-truth previous words)
    # -> can pass entire ground-truth trg sentence at once, so uses PackedSequences.
    def forward(self, decoder_inputs, encoder_states, initial_state):
        if not self.training:
            return self.predict(decoder_inputs, encoder_states, initial_state)
        else:
            embs = self.embed(decoder_inputs["in"]) # (bsz x max_trg_len x input_size)
            packed_input = pack_padded_sequence(embs, decoder_inputs["lengths"], batch_first=True)
            packed_output, _ = self.lstm(packed_input, initial_state) # (total_len x hidden_size)
            # (where total_len is the total number of non-pad tokens in the batch).
            
            if self.attention:
                decoder_states, _ = pad_packed_sequence(packed_output, batch_first=True) # (bsz x max_trg_len x hidden_size)
                # dropout applied prior to computing attention.
                # if self.add_drop_layer:
                #     decoder_states = self.dropout_layer(decoder_states)
                #### added #####
                bsz = decoder_states.size(0)
                out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, self.hidden_size).cuda(), p=self.out_drop, training=self.training)
                decoder_states = decoder_states * out_mask
                ################
                attention_states = self.attend(decoder_states, encoder_states, decoder_inputs["mask"]) # (bsz x max_trg_len x 2*hidden_size)
                attention_states = pack_padded_sequence(attention_states, decoder_inputs["lengths"], batch_first=True).data # (total_len x 2*hidden_size)
            else:
                attention_states = packed_output.data # (total_len x hidden_size)
                # if self.add_drop_layer:
                #     attention_states = self.dropout_layer(attention_states)
                #### added #####
                bsz = attention_states.size(0)
                out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, self.hidden_size).cuda(), p=self.out_drop, training=self.training)
                attention_states = attention_states * out_mask
                ################

            del packed_input, packed_output
            if self.project_att_states:
                #att_layer_states = F.relu(self.attention_layer(attention_states)) # (total_len x input_size) if tie_weights. else (total_len x hidden_size)
                ### change to tanh ###
                att_layer_states = torch.tanh(self.attention_layer(attention_states)) # (total_len x input_size) if tie_weights. else (total_len x hidden_size)
            else:
                att_layer_states = attention_states # (total_len x hidden_size)

            dists = self.out(att_layer_states) # (total_len x vocab_size)
            # -> for each sequence of batch, for each time step,
            # holds predicted logits over next words.

            return dists


    # -returns concatenation of the queries with their context vectors
    # (bsz x (2*hidden_size)).
    # -queries (decoder states) is (bsz x max_trg_len x decoder_hidden_size).
    # -keys (encoder states) is (bsz x max_src_len x encoder_hidden_size).
    # -mask is (bsz x 1 x max_src_len).
    # mask[i][0][k] = 1 wherever keys[i][k] holds hidden state corresponding
    # to a pad token (i.e., does not correspond to actual src token
    # so dn include it in the weighted average).
    # -each of the max_trg_len decoder states in queries
    # computes attention over same set of encoder states, so dim 1 of mask
    # broadcasts over each query).
    def attend(self, queries, keys, mask):
        # entry i,j,k holds dot product of query j (hidden state of decoder timestep j)
        # with key k (hidden state of encoder timestep k), for i'th sequence of batch.
        scores = torch.bmm(queries, keys.transpose(1,2)) # (bsz x max_trg_len x max_src_len)
        if self.scaled:
            # larger hidden sizes produce dot products of greater magnitude.
            # normalizing scores provides smoother softmax results.
            scores.true_divide_(self.sqrt_hs) 
        # everywhere mask holds a 1, fill the corresponding entry of scores with negative infinity.
        # -> this overwrites the dot products with encoder states of pad tokens, 
        # so when apply softmax, their weight gets set to zero.
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores, dim=2)
        # context vector for a given query is an average over all encoder states,
        # weighted by their dot product with the query.
        contexts = torch.bmm(weights, keys) # (bsz x max_trg_len x hidden_size)
        del scores, weights

        # concatenate contexts with the original decoder states
        attention_states = torch.cat((contexts, queries), dim=2) # (bsz x max_trg_len x 2*hidden_size)

        return attention_states


    def set_inference_alg(self, inference_alg="greedy_search"):
        self.inference_alg = inference_alg
        if inference_alg == "beam_search":
            self.predict = self.beam_search_translate
        elif inference_alg == "greedy_search": 
            self.predict = self.greedy_search_translate
        else:
            raise NameError(f"tried to set an unsupported inference algorithm: {inference_alg}. see readme for valid inference_alg options.")


    # -applied to multiple sequences in parallel, for each of which it
    # predicts entire beam in parallel.
    # -each beam's sequences maintained in descending order by likelihood.
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
        max_src_len = decoder_inputs["max_src_len"]
        mask = decoder_inputs["mask"]
        eos = torch.tensor([self.eos_idx], dtype=torch.long, device=self.device)
        T = max_src_len + self.decode_slack # max number of decoder time steps
        # whenever a beam's most probable sequence produces eos for the
        # first time, copy it to corresponding row of translation.
        translation = torch.zeros((bsz, T), dtype=torch.long, device=self.device)
        # stop when most likely sequence of each beam has predicted the eos token.
        finished = torch.zeros((bsz,), dtype=torch.bool, device=self.device)
        # -> entry j is 1 if have finished translating sentence j.

        # handle timestep 1 separately from the rest, bc initializes the beams 
        # (employs a beam_width of 1, rather than b), each of which was produced
        # by its src sequence's final hidden state.
        sos_tokens = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device)
        input_0 = self.embed(sos_tokens) # (bsz x 1 x input_size)
        h_0, c_0 = initial_state
        dists_1, (h_1, c_1) = self.decode_step(input_0, (h_0, c_0), mask, encoder_states) 
        # -> dists_1 is (bsz x v), h_1 and c_1 are (nl x bsz x hidden_size)
        sequences, seq_likelihoods, (h_i, c_i), top_words = initialize_beams(dists_1, (h_1, c_1), hp)
        input_i = self.embed(top_words.view(bsz*b, 1)) # (bsz*b x 1 x input_size)

        ### edge case: immediately predicts eos (only plausible in early stages of overfitting during unit tests)
        write_finished_translations(translation, sequences, finished, eos, 1)
        if finished.all():
            return translation
        #########################

        # for a given src_seq of batch, its encoder_states are attended to by
        # each of the b sequences in the beam for the corresponding trg_seq:
        expanded_encoder_states = encoder_states.repeat(1,b,1).view(bsz*b,max_src_len,self.hidden_size)
        expanded_mask = mask.repeat(1,b,1).view(bsz*b,1,max_src_len)

        # timesteps 2 and onward handled inside loop.
        # iter i takes sequences of length i-1 and extends them to length i.
        for i in range(2, T):
            dists_i, (h_i, c_i) = self.decode_step(input_i, (h_i, c_i), expanded_mask, expanded_encoder_states) # (bsz*b x v)
            # each of the b sequences of a beam generates b successors,
            # providing b*b candidates for the next beam.
            candidate_likelihoods, top_words = expand_beams(dists_i, seq_likelihoods, hp)
            # identify the top b most likely of the b*b candidates.
            top_candidate_likelihoods, top_candidates = torch.topk(candidate_likelihoods, b) # (bsz x b)
            seq_likelihoods = top_candidate_likelihoods # update seq likelihoods
            next_words = torch.gather(top_words, 1, top_candidates) # words appended to beam sequences
            # fill beams in descending order with most likely sequences,
            # along with the hidden states that produced them.
            sequences, (h_i, c_i) = update_beams(sequences, next_words, top_candidates, (h_i, c_i), i, hp)
            # must write a translation as soon as it's finished, bc if
            # continue extending it, resulting sequence will be of extremely
            # low probability (words never follow <eos>) and therefore lost.
            write_finished_translations(translation, sequences, finished, eos, i)
            if finished.all():
                return translation

            input_i = self.embed(next_words.view(bsz*b, 1)) # (bsz*b x 1 x input_size)

        ### edge case: copy the translations whose beams never produced a
        # most-likely sequence ending in eos.
        never_finished_indices = torch.nonzero(finished.logical_not(), as_tuple=False).squeeze(1)# (num_never_finished, )
        for j in never_finished_indices:
            translation[j,:i] = sequences[j][0]
        #########################

        # (for each translation, will extract the portion prior to the eos symbol
        # inside postprocess.py)
        return translation

    
    # -compute the decoder states one at a time, using prediction from previous
    # time step as input to current timestep.
    # -returns (bsz x T) tensor, where T is number of decode time steps
    # (at least 1, and at most max_src_len + decode_slack),
    # holding the batch of predicted translations.
    def greedy_search_translate(self, decoder_inputs, encoder_states, initial_state):
        bsz = encoder_states.size(0)
        eos = torch.tensor([self.eos_idx], dtype=torch.long, device=self.device)
        # -length of longest src sentence of dev batch, along with decode_slack
        # used to bound the max number of decoding timesteps
        # (heuristic that a trg sent will have similar number of words as its
        # src sent. decode_slack is how many more it can have, before the
        # translation "cuts off").
        T = decoder_inputs["max_src_len"] + self.decode_slack
        decoder_in = torch.full((bsz, 1), self.sos_idx, dtype=torch.long, device=self.device) # (bsz x 1)
        # initialize running translation (each decode step concatenates to it)
        translation = torch.tensor([], dtype=torch.long, device=self.device)
        # stop decoding when every sentence has predicted the eos token.
        finished = torch.zeros((bsz,), dtype=torch.bool, device=self.device)
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
    # fwd pass for batch of length 1 sentences, corresponding to time step i.
    def decode_step(self, input_i, hidden, mask, encoder_states):
        output_i, (h_i, c_i) = self.lstm(input_i, hidden) # output_i is (bsz x 1 x hidden_size)
        if self.attention:
            attention_states_i = self.attend(output_i, encoder_states, mask).squeeze(1)
            # -> (bsz x 2*hidden_size)
        else:
            attention_states_i = output_i.squeeze(1) # (bsz x hidden_size)
        if self.project_att_states:
            att_layer_states_i = F.relu(self.attention_layer(attention_states_i)) 
        else:
            att_layer_states_i = attention_states_i

        dists_i = self.out(att_layer_states_i) # (bsz x vocab_size)

        return dists_i, (h_i, c_i)