import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, hyperparams):
        super(Encoder, self).__init__()
        # hyperparameter settings
        self.vocab_size = hyperparams["src_vocab_size"]
        self.input_size = hyperparams["enc_input_size"]
        self.hidden_size = hyperparams["enc_hidden_size"]
        self.num_layers = hyperparams["enc_num_layers"]
        self.decoder_num_layers = hyperparams["dec_num_layers"] # final_to_first_initializer needs to know how many layers decoder has
        self.lstm_dropout = hyperparams["enc_lstm_dropout"]
        self.add_drop_layer = hyperparams["enc_dropout"] > 0
        self.out_drop = hyperparams["enc_dropout"]
        self.attention = hyperparams["attention_fn"] != "none"
        self.bidirectional = hyperparams["bidirectional"]
        self.reverse_src = hyperparams["reverse_src"]
        self.device = hyperparams["device"]
        if hyperparams["decoder_init_scheme"] == "layer_to_layer":
            self.initialize_decoder_state = self.layer_to_layer_initializer
        elif hyperparams["decoder_init_scheme"] == "final_to_first":
            self.initialize_decoder_state = self.final_to_first_initializer
        else:
            raise NameError(f"specified an unsupported decoder init scheme: {hyperparams['decoder_init_scheme']}")

        # architecture
        self.embed = nn.Embedding(self.vocab_size, self.input_size)
        ### added:
        #self.embed.weight.data.uniform_(-.1, .1)



        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.lstm_dropout, batch_first=True)
        #if self.add_drop_layer:
        #    self.dropout_layer = nn.Dropout(p=hyperparams["enc_dropout"])
        if self.bidirectional:
            # uses separate sets of parameters to construct the states to compute attention with,
            # and the state for initializing the decoder hidden state.
            self.bridge = nn.Linear(2*self.hidden_size, self.hidden_size)
            if self.attention:
                self.project_keys = nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, encoder_inputs):
        embs = self.embed(encoder_inputs["in"])
        packed_input = pack_padded_sequence(embs, encoder_inputs["sorted_lengths"], batch_first=True)
        packed_output, (hn, cn) = self.lstm(packed_input)
        encoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
        # -> encoder_states is 3D tensor of size (bsz x max_src_len x num_directions*encoder_hidden_size)
        # dropout applied prior to projecting back to decoder hidden size.
        #if self.add_drop_layer:
        #    encoder_states = self.dropout_layer(encoder_states)
        ### added ###
        bsz = encoder_states.size(0) # varies for "remainder" batch
        d_hid = encoder_states.size(2) # varies if bidirectional
        out_mask = F.dropout(torch.ones(bsz, 1, d_hid).cuda(), p=self.out_drop, training=self.training)
        encoder_states = encoder_states * out_mask
        ############

        del packed_input, packed_output
        initial_h, initial_c = self.initialize_decoder_state(hn, cn)
        # -> initial_h and initial_c are each 3D tensors of size (decoder_num_layers x bsz x decoder_hidden_size)

        if self.bidirectional and self.attention:
            # project encoder_states back to decoder_hidden_size, so can apply attention
            #encoder_states = F.relu(self.project_keys(encoder_states)) 
            ### remove nonlinearity ###
            encoder_states = self.project_keys(encoder_states)
            ########

        # unsorting step:
        #!!!change so that only needs to do this when in training mode.
        idxs_in_sorted = encoder_inputs["idxs_in_sorted"]
        initial_h = initial_h[:,idxs_in_sorted]
        initial_c = initial_c[:,idxs_in_sorted]
        encoder_states = encoder_states[idxs_in_sorted]
        
        return encoder_states, (initial_h, initial_c)


    # init layer i of decoder hidden state (to be used in first decoder timestep)
    # with layer i of encoder hidden state from final encoder time-step.
    def layer_to_layer_initializer(self, hn, cn):
        if not self.bidirectional:
            initial_h, initial_c = hn, cn
        else:
            # -transform encoder hidden states (from each direction)
            # of final time step into a single initial decoder hidden state.
            # -extract hidden and memory cell states of each layer,
            # from each direction, for final time step.
            final_fwd_h, final_bwd_h = hn[0:hn.size(0):2], hn[1:hn.size(0):2] 
            ### changed: ###
            #final_fwd_c, final_bwd_c = cn[0:cn.size(0):2], cn[1:cn.size(0):2]
            ################
            # -> each is (num_layers x bsz x hidden_size)

            # concatenate directions.
            initial_h = torch.cat((final_fwd_h, final_bwd_h), dim=2) 
            ###### changed ######
            #initial_c = torch.cat((final_fwd_c, final_bwd_c), dim=2)
            ############
            # -> each is (num_layers x bsz x 2*hidden_size)

            # project back to hidden_size.
            # initial_h = F.relu(self.bridge(initial_h)) 
            # initial_c = F.relu(self.bridge(initial_c))

            ### change bridge to tanh, and do not pass memory cell ###
            initial_h = torch.tanh(self.bridge(initial_h)) 
            initial_c = torch.zeros_like(initial_h)
            ###############
            # -> each is (num_layers x bsz x hidden_size)

        return initial_h, initial_c


    # initialize only the first layer of decoder hidden state
    # (to be used in first decoder time-step) with final layer of
    # encoder hidden state from final encoder time-step.
    # (all non-first decoder hidden layers initialized as zeros).
    def final_to_first_initializer(self, hn, cn):
        initial_h = torch.zeros(self.decoder_num_layers, hn.size(1), self.hidden_size, device=self.device)
        initial_c = torch.zeros(self.decoder_num_layers, cn.size(1), self.hidden_size, device=self.device)
        if not self.bidirectional:
            top_layer_h, top_layer_c = hn[-1], cn[-1]
            # -> each is (bsz x hidden_size)
        else:
            # extract hidden and memory cell states of final layer,
            # from each direction, for final time step. 
            top_layer_fwd_h, top_layer_bwd_h = hn[-2], hn[-1]
            top_layer_fwd_c, top_layer_bwd_c = cn[-2], cn[-1] 
            # -> each is (bsz x hidden_size)

            # concatenate directions.
            # NOTE: dim is 1, not 2, in this init scheme
            top_layer_h = torch.cat((top_layer_fwd_h, top_layer_bwd_h), dim=1) 
            top_layer_c = torch.cat((top_layer_fwd_c, top_layer_bwd_c), dim=1)
            # -> each is (bsz x 2*hidden_size)

            # project back to hidden_size.
            top_layer_h = F.relu(self.bridge(top_layer_h)) 
            top_layer_c = F.relu(self.bridge(top_layer_c)) 
            # -> each is (bsz x hidden_size)

        initial_h[0], initial_c[0] = top_layer_h, top_layer_c

        return initial_h, initial_c