import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from random import shuffle

from processCorpuses import formatPrediction, naiveRecase



class RNNencdec(nn.Module):
    def __init__(self, encoder, decoder, embType):
        super(RNNencdec, self).__init__()
        if encoder.bi_enc and not encoder.project:
            assert decoder.hidden_size == 2 * encoder.hidden_size
        else:    
            assert decoder.hidden_size == encoder.hidden_size
            #assert decoder.hidden_size == decoder.input_size
        assert decoder.num_layers == encoder.num_layers
        #assert decoder.num_layers == 1
        assert encoder.dev == decoder.dev
        self.encoder = encoder
        self.decoder = decoder
        if embType == 'jointBPE': # ??share vocabs in enc and dec?? what if tie weights to projectToV as well??
            self.encoder.src_embeddings.weight = self.decoder.trg_embeddings.weight
        
    
    def forward(self, encoder_inputs_batch, decoder_inputs_batch):
        encoder_states, decoder_initial_state, src_lengths = self.encoder(encoder_inputs_batch)
        return self.decoder(decoder_inputs_batch, encoder_states, decoder_initial_state, src_lengths)


class Encoder(nn.Module):
    def __init__(self, encoder_params):
        super(Encoder, self).__init__()
        self.vocab_size = encoder_params['vocab_size']
        self.input_size = encoder_params['input_size']
        self.hidden_size = encoder_params['hidden_size']
        self.num_layers = encoder_params['num_layers']
        self.dropout = encoder_params['dropout']
        self.dev = encoder_params['dev']
        self.bi_enc = encoder_params['bi_enc']
        self.project = encoder_params['project']
        self.reverse_src = encoder_params['reverse_src']
        self.init_scheme = encoder_params['init_scheme']
        self.customLSTM = encoder_params['customLSTM']

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        # no longer bidirectional, bc complicates hp settings with WT
        #self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, bidirectional=self.bi_enc, batch_first=True)
        #self.weights = encoder_params['weights_to_drop']
        if not self.customLSTM:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bi_enc, dropout=self.dropout, batch_first=True)
            #self.init_lstm_dc()
        else:
            self.lstm0 = nn.LSTM(self.input_size, self.hidden_size, 1, bidirectional=self.bi_enc, batch_first=True)
            self.lstm1 = nn.LSTM((2 if self.bi_enc else 1) * self.hidden_size, self.hidden_size, 1, bidirectional=self.bi_enc, batch_first=True)
        
        if self.bi_enc:
            self.bridge = nn.Linear(2*self.hidden_size, self.hidden_size)
            self.projKeys = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.src_embeddings = nn.Embedding(self.vocab_size, self.input_size)
        self.src_embeddings.weight.data.uniform_(-.1, .1)
        #self.emb_drop = nn.Dropout(encoder_params['src_emb_drop'])
        #self.dc = encoder_params['dropconnect']

        #self.variational_drop = nn.Dropout(encoder_params['variational_drop'])
        #self.i0_drop = encoder_params['i0_drop']
        #self.i1_drop = encoder_params['i1_drop']
        self.out_drop = encoder_params['out_drop']
        self.tanh = nn.Tanh()

    # init lstm for use in dropconnect (weightdrop)
    def init_lstm_dc(self):
        for name_w in self.weights:
            w = getattr(self.lstm, name_w)
            self.lstm.register_parameter(name_w + '_raw', torch.nn.Parameter(w.data))
        self.lstm.flatten_parameters()


    def myWeightDrop(self):
        for name_w in self.weights:
            raw_w = getattr(self.lstm, name_w + '_raw')
            w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dc, training=self.training))
            setattr(self.lstm, name_w, w)

        self.lstm.flatten_parameters()
      


    def myEmbDrop(self, batch):
        mask = self.emb_drop(torch.ones(len(self.vocab), 1).cuda())
        masked_embeddings_weight = mask * self.src_embeddings.weight
        masked_embs = torch.nn.functional.embedding(batch, masked_embeddings_weight)
        
        return masked_embs


    def myVarInf(self, batch, drop=.5): # receives either a batch of embeddings or a batch of lstm outputs
        # batch is (bsz x seq_len x X), where X is either d_emb or d_hid
        bsz, seq_len, dm = batch.size()
        mask = torch.nn.functional.dropout(torch.ones(bsz, 1, dm).cuda(), p=drop, training=self.training)
        return mask.expand(bsz, seq_len, dm) * batch




    def forward(self, encoder_inputs_batch):
        encoder_inputs_tensor = encoder_inputs_batch[0]
        src_lengths = encoder_inputs_batch[1]
        idxs_in_sorted_enc_inputs_tensor = encoder_inputs_batch[2]

        srcEmbeddings = self.src_embeddings(encoder_inputs_tensor)
        #srcEmbeddings = self.myEmbDrop(encoder_inputs_tensor)
        #srcEmbeddings = self.myVarInf(srcEmbeddings, self.i0_drop)
        # srcEmbeddings is a 3D tensor of shape (bsz x max_len x input_size)
        # packed_input is a tuple, whose 1st comp is the packed sequence, and 2nd comp is "batchsizes",
        # which holds, at position i, the number of inputs for time step i 

        packed_input = pack_padded_sequence(srcEmbeddings, src_lengths, batch_first=True)
        #self.myWeightDrop()
        #packed_output, encoder_final = self.lstm(packed_input)
        #del packed_input
        # packed_input = pack_padded_sequence(srcEmbeddings, src_lengths, batch_first=True)
        # if self.customLSTM:
        #     ### pass thru layer 0 lstm ###
        #     packed_output, (enc0_hn, enc0_cn) = self.lstm0(packed_input) # output is (bsz x seq_len x d_hid)
        #     del packed_input
            
        #     # encoder_states is bsz x seq_len x d_hid; contains d_hid for final (only) layer
        #     # hn_0 is nl x bsz x d_hid
        #     # bc each layer is its own lstm of 1 layer:
        #     # encoder_states[:,-1,:] == hn_0[0]
        #     # therefore, so that same dropout mask is applied to what will be
        #     # the input to lstm1, and the encoding that will be the input to decoder,
        #     # initialize decoder using encoder_states instead of enc0_hn
        #     encoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
        #     del packed_output
        #     encoder_states = self.myVarInf(encoder_states, self.i1_drop)
        #     dec0_h0, dec0_c0 = encoder_states[:,-1,:].unsqueeze(0), enc0_cn
        #     del enc0_hn, enc0_cn

        #     ### pass thru layer 1 lstm ###
        #     packed_input = pack_padded_sequence(encoder_states, src_lengths, batch_first=True)
        #     packed_output, (enc1_hn, enc1_cn) = self.lstm1(packed_input) # output is (bsz x seq_len x d_hid)
        #     del packed_input
        #     encoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
        #     del packed_output
        #     # encoder_states = self.myVarInf(encoder_states, self.out_dr)
        #     dec1_h0, dec1_c0 = enc1_hn, enc1_cn
        #     del enc1_hn, enc1_cn
        #     # unsort all encodings and encoder_states and src_lengths
        #     dec0_h0 = dec0_h0[:,idxs_in_sorted_enc_inputs_tensor]
        #     dec0_c0 = dec0_c0[:,idxs_in_sorted_enc_inputs_tensor]

        #     dec1_h0 = dec1_h0[:,idxs_in_sorted_enc_inputs_tensor]
        #     dec1_c0 = dec1_c0[:,idxs_in_sorted_enc_inputs_tensor]

        #     encoder_states = encoder_states[idxs_in_sorted_enc_inputs_tensor]
        #     # unsorted src lengths used by attention mechanism for masking
        #     src_lengths = src_lengths[idxs_in_sorted_enc_inputs_tensor] 
        #     # instead, returns a list of initial decoder states
        #     return encoder_states, [(dec0_h0, dec0_c0), (dec1_h0, dec1_c0)], src_lengths



        packed_output, (hn, cn) = self.lstm(packed_input)
        encoder_states, _ = pad_packed_sequence(packed_output, batch_first=True) # 3D tensor of size (bsz x max_len x 2*enc_hs)
        del packed_output

        bsz = encoder_states.size(0) # varies for "remainder" batch
        d_hid = encoder_states.size(2) # varies if bidirectional
        out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, d_hid).cuda(), p=self.out_drop, training=self.training)
        encoder_states = encoder_states * out_mask


        if self.bi_enc:
            encoder_states = self.projKeys(encoder_states) # project to dim of decoder hs
        
            # initialize only the initial hidden state, not the memory cell
            finalForwardStates = hn[0:hn.size(0):2] # nl x bsz x d_hid
            finalBackwardStates = hn[1:hn.size(0):2] # nl x bsz x d_hid
            decoder_initial_h = torch.cat((finalForwardStates, finalBackwardStates), dim=2) # nl x bsz x 2*d_hid
            del finalForwardStates, finalBackwardStates
            #decoder_initial_h = self.projKeys(decoder_initial_h) # nl x bsz x d_hid
            decoder_initial_h = self.tanh(self.bridge(decoder_initial_h)) # nl x bsz x d_hid

            decoder_initial_c = torch.zeros_like(decoder_initial_h)

        else:
            decoder_initial_h, decoder_initial_c = hn, cn

        #encoder_states = self.myVarInf(encoder_states, self.out_drop)
        ###could also apply dropout to the memory cell, but leaving out for now
        #decoder_initial_h, decoder_initial_c = self.initializeDecoderHidden(encoder_states, encoder_final)
        #decoder_initial_h, decoder_initial_c = self.initializeDecoderHidden(encoder_final)
        # each are 3D tensors of size (dec_nl x bsz x dec_hs)
        
        # encoder_states serves as the keys/values for attention
        # (decoder_initial_h, decoder_initial_c) serves as encoding that initializes decoder
        # each needs to be unsorted so that align correctly with corresponding trg sentences
        decoder_initial_h = decoder_initial_h[:,idxs_in_sorted_enc_inputs_tensor]
        decoder_initial_c = decoder_initial_c[:,idxs_in_sorted_enc_inputs_tensor]
        decoder_initial_state = (decoder_initial_h, decoder_initial_c)

        # unsorted, padded encoder_states
        #padded_encoder_states = encoder_states[idxs_in_sorted_enc_inputs_tensor]
        encoder_states = encoder_states[idxs_in_sorted_enc_inputs_tensor]
        # unsorted src lengths used by attention mechanism for masking
        src_lengths = src_lengths[idxs_in_sorted_enc_inputs_tensor] 

        return encoder_states, decoder_initial_state, src_lengths





    def initializeDecoderHidden(self, encoder_final):
        bsz = encoder_final[0].size()[1]
        #!!!hacky workaround: assume a decoder of 1 layer
        #decoder_initial_h = torch.zeros(1, bsz, self.hidden_size, device=self.dev)
        #decoder_initial_c = torch.zeros(1, bsz, self.hidden_size, device=self.dev)

        if self.init_scheme == "layer_to_layer":
            return encoder_final

        elif self.init_scheme == "final_to_first":
            decoder_initial_h = torch.zeros(self.num_layers, bsz, self.hidden_size, device=self.dev)
            decoder_initial_c = torch.zeros(self.num_layers, bsz, self.hidden_size, device=self.dev)
            # initialize first layer of decoder with last layer of encoder
            decoder_initial_h[0] = encoder_final[0][-1] 
            decoder_initial_c[0] = encoder_final[1][-1] 

        # initialize every layer of decoder with last layer of encoder
        #elif self.init_scheme == "final_to_every":
        #decoder_initial_h = encoder_final[0][-1].unsqueeze(0).expand(self.num_layers, bsz, self.hidden_size)
        #decoder_initial_c = encoder_final[1][-1].unsqueeze(0).expand(self.num_layers, bsz, self.hidden_size)

        return (decoder_initial_h, decoder_initial_c)




class Decoder(nn.Module):
    def __init__(self, decoder_params):
        super(Decoder, self).__init__()
        self.vocab_size = decoder_params['vocab_size']
        self.eosIdx = decoder_params['eosIdx']
        self.sosIdx = decoder_params['sosIdx']
        self.padIdx = decoder_params['padIdx']

        self.idx_to_trg_word = decoder_params['idx_to_trg_word']
        self.input_size = decoder_params['input_size']
        self.hidden_size = decoder_params['hidden_size']
        #assert self.input_size == self.hidden_size / 2 #!new, for WT
        self.num_layers = decoder_params['num_layers']
        self.dropout = decoder_params['dropout']
        self.attention = decoder_params['attention']
        if self.attention == "scaled_dot_product_attn":
            self.sqrt_hs = math.sqrt(self.hidden_size) # so do not need to recalc in each attn step
        else:
            self.sqrt_hs = 1. # so that scale step of attn is a no-op
        self.inf_alg = decoder_params['inf_alg'] # inference algorithm
        self.beam_size = decoder_params['beam_size']
        self.decode_slack = decoder_params['decode_slack']
        self.dev = decoder_params['dev']
        self.tf = True

        #self.weights = decoder_params['weights_to_drop']
        #self.emb_drop = nn.Dropout(decoder_params['trg_emb_drop'])
        #self.variational_drop = nn.Dropout(decoder_params['variational_drop'])

        #self.i0_drop = decoder_params['i0_drop']
        #self.i1_drop = decoder_params['i1_drop']
        self.out_drop = decoder_params['out_drop']
        #self.dc = decoder_params['dropconnect']
        #self.att_drop = decoder_params['att_drop']

        self.customLSTM = decoder_params['customLSTM']
        #self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)
        if not self.customLSTM:
            self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)
            #self.init_lstm_dc()
        else:
            self.lstm0 = nn.LSTM(self.input_size, self.hidden_size, 1, batch_first=True)
            self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)

        
        

        self.trg_embeddings = nn.Embedding(self.vocab_size, self.input_size)

        

        self.softmax1 = nn.Softmax(dim=1) 
        self.softmax2 = nn.Softmax(dim=2) 
        self.logsoftmax1 = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

        #new!!! TIE THE WEIGHTS. decoder_hidden must equal decoder input_size
        if decoder_params['tie_weights']:
            self.projectToV = nn.Linear(self.input_size, self.vocab_size) 
            #assert self.hidden_size == self.input_size
            self.projectToV.weight = self.trg_embeddings.weight
            self.init_weights()
            if self.attention != None:
                self.att_layer = nn.Linear(2*self.hidden_size, self.input_size) 
        
        else:
            self.projectToV = nn.Linear(self.hidden_size, self.input_size) 
            self.trg_embeddings.weight.data.uniform_(-.1, .1)
            if self.attention != None:
                self.att_layer = nn.Linear(2*self.hidden_size, self.hidden_size) 
        
    #!!!new init scheme and WT
    def init_weights(self):
        initrange = 0.1
        self.trg_embeddings.weight.data.uniform_(-initrange, initrange)
        self.projectToV.bias.data.zero_()
        


    # init lstm for use in dropconnect (weightdrop)
    def init_lstm_dc(self):
        for name_w in self.weights:
            w = getattr(self.lstm, name_w)
            self.lstm.register_parameter(name_w + '_raw', torch.nn.Parameter(w.data))
        self.lstm.flatten_parameters()



    def myWeightDrop(self):
        for name_w in self.weights:
            raw_w = getattr(self.lstm, name_w + '_raw')
            w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dc, training=self.training))
            setattr(self.lstm, name_w, w)

        self.lstm.flatten_parameters()



    def myEmbDrop(self, batch):
        mask = self.emb_drop(torch.ones(len(self.vocab), 1).cuda())
        masked_embeddings_weight = mask * self.trg_embeddings.weight
        masked_embs = torch.nn.functional.embedding(batch, masked_embeddings_weight)
        
        return masked_embs


    def myVarInf(self, batch, drop=.5): # receives either a batch of embeddings or a batch of lstm outputs
        # batch is (bsz x seq_len x X), where X is either d_emb or d_hid
        bsz, seq_len, dm = batch.size()
        mask = torch.nn.functional.dropout(torch.ones(bsz, 1, dm).cuda(), p=drop, training=self.training)
        return mask.expand(bsz, seq_len, dm) * batch
        

    def forward(self, decoder_inputs_batch, padded_encoder_states, decoder_initial_state, src_lengths):
        if not self.tf: # not teaching forcing (making prediction for a single sentence instead)
            # don't need the input (decoder_inputs_batch = None), so skip its preprocessing steps
            if self.inf_alg == "greedy_search":
                return self.getGreedyTranslation(decoder_inputs_batch, padded_encoder_states, decoder_initial_state, src_lengths)
            elif self.inf_alg == "beam_search":
                return self.getBeamSearchTranslation(decoder_inputs_batch, padded_encoder_states, decoder_initial_state, src_lengths)

        else:
            # teacher forcing (perform training step on the batch)
            decoder_inputs_tensor = decoder_inputs_batch[0]
            trg_lengths = decoder_inputs_batch[1]
            mask = decoder_inputs_batch[2]
            #bsz = decoder_inputs_tensor.size(0)

            trgEmbeddings = self.trg_embeddings(decoder_inputs_tensor) # 3D tensor of shape (bsz x max_len x input_size)
            #trgEmbeddings = self.myEmbDrop(decoder_inputs_tensor)
            #trgEmbeddings = self.myVarInf(trgEmbeddings, self.i0_dr)

            packed_input = pack_padded_sequence(trgEmbeddings, trg_lengths, batch_first=True)

            # packed_input is a tuple, whose first comp is the packed sequence, and second comp is "batchsizes",
            # which holds, at position i, the number of inputs for time step i 
            return self.computeProbDists(packed_input, padded_encoder_states, decoder_initial_state, src_lengths, trg_lengths, mask)
        

    # model makes its predictions for next words given the actual previous words (teacher-forcing)
    def computeProbDists(self, packed_input, padded_encoder_states, decoder_initial_state, src_lengths, trg_lengths, mask):
        
        # if self.customLSTM:
        #     (dec0_h0, dec0_c0) = decoder_initial_state[0]
        #     (dec1_h0, dec1_c0) = decoder_initial_state[1]
        #     packed_output, _ = self.lstm0(packed_input, (dec0_h0, dec0_c0))
        #     del packed_input
        #     decoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
        #     del packed_output
        #     decoder_states = self.myVarInf(decoder_states, drop=self.i1_drop)
        #     packed_input = pack_padded_sequence(decoder_states, trg_lengths, batch_first=True)
        #     packed_output, _ = self.lstm1(packed_input, (dec1_h0, dec1_c0))
        #     del packed_input

        # else:
        #self.myWeightDrop()

        packed_output, _ = self.lstm(packed_input, decoder_initial_state)
        del packed_input
        





        # Let q be total length of all trg seqs of the batch. then packed_decoder_states is 2D tensor of shape (q x hs)
        if self.attention != None:
            # convert states to attentional states before projecting
            decoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
            del packed_output
            bsz = decoder_states.size(0)


            out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, self.hidden_size).cuda(), p=self.out_drop, training=self.training)
            decoder_states = decoder_states * out_mask

            #decoder_states = self.myVarInf(decoder_states, self.out_drop)
            packed_decoder_states = self.get_attStates(decoder_states, padded_encoder_states, trg_lengths, mask)
        else:
            packed_decoder_states = packed_decoder_states.data
        
        del decoder_states
        distsOverNextWords = self.logsoftmax1(self.projectToV(packed_decoder_states.data)) # distsOverNextWords is of shape (q x V_trg)
        # convert each dist into a probdist

        return distsOverNextWords









    # parallelized version
    def getGreedyTranslation(self, decoder_inputs_batch, encoder_states, decoder_initial_state, src_lengths):
        sosIdx = self.sosIdx
        eosIdx = self.eosIdx
        bsz = encoder_states.size(0)
        decoder_inputs_tensor = torch.full((bsz, 1), sosIdx).long().cuda() # (bsz x 1)
        trg_lengths = torch.full((bsz,), 1).long().cuda()

        mask = decoder_inputs_batch[0]
        corpus_indices = decoder_inputs_batch[1]
        max_src_len = src_lengths.max()    
        translation = torch.tensor([]).long().cuda() # initialization for what will be (bsz x numTranslatedWordsSoFar)

        
        
        input_i = self.trg_embeddings(decoder_inputs_tensor) # (bsz x 1 x input_size)

        max_src_len = src_lengths.max()
        (h_i, c_i) = decoder_initial_state
        # if customLSTM, then decoder_initial_state is a list
        #(h_i0, c_i0) = decoder_initial_state[0] # (nl x bsz x hs, nl x bsz x hs)
        #(h_i1, c_i1) = decoder_initial_state[1] # (nl x bsz x hs, nl x bsz x hs)

        # compute the decoder states one at a time:
        # (set some sanity check to cut off overly long translations)
        for i in range(max_src_len + self.decode_slack): 
            ###!!change this to acct for 2 lstms
            output, (h_i, c_i) = self.lstm(input_i, (h_i, c_i))
            #output, (h_i0, c_i0) = self.lstm0(input_i, (h_i0, c_i0))
            #output, (h_i1, c_i1) = self.lstm1(output, (h_i1, c_i1)) # bsz x seq_len x d_hid
            # this lstm is unidirectional, so h_i is 3D tensor of shape:
            # (nl * nd x bsz x hs) = (nl x bsz x hs)
            # only want to project the output of the last layer:
            #decoder_state = h_i[-1] # 2D tensor of size (bsz x hs)
            #decoder_state = output[:,-1,:] # 2D tensor of size (bsz x hs)

            if self.attention != None:
                # get_attStates() expects a 3D tensor of size (bsz x trg_max_len x hs) of decoder states,
                # so reshape to accomodate hidden state for a single time step
                #decoder_state = self.get_attStates(decoder_state.unsqueeze(1), encoder_states, src_lengths, trg_lengths, mask)
                decoder_state = self.get_attStates(output, encoder_states, trg_lengths, mask)

                # get_attStates() returns (q x hs), which is (bsz x hs), in this case, so of correct shape

            # (bsz x hs) x (hs x V_trg) = (bsz x V_trg):
            distOverNextWords = self.projectToV(decoder_state.data)
            # normalize into a probability distribution over possible next words:
            distOverNextWords = self.logsoftmax1(distOverNextWords) 

            # greedy search: take argmax to get position of dist containing highest probability value
            indexOfMostLikelyWord = torch.argmax(distOverNextWords,1).unsqueeze(1) # 2D of size bsz x 1
    
            # concat to the running translations for each seq of the batch
            translation = torch.cat((translation, indexOfMostLikelyWord), dim=1)
            
            # get embeddings, so can pass as next input
            input_i = self.trg_embeddings(indexOfMostLikelyWord) # (bsz x 1 x input_size)


        # translation is now (bsz x max_src_len+decode_slack)
        tlist = translation.tolist()
        del translation
        translations = []
        for j in range(len(tlist)):
            try:
                eosIndex = tlist[j].index(eosIdx) # keep translation up to but not including eos
            except ValueError:
                eosIndex = len(tlist[j]) # never produced eos, so keep entire translation
            
            list_trans = [self.idx_to_trg_word[idx] for idx in tlist[j][:eosIndex]]
            try:
                #list_trans = naiveRecase(list_trans)
                #jth_pred = ' '.join(list_trans)
                jth_pred = formatPrediction(list_trans)

            except IndexError:
                print("predicted eos as first word")
                jth_pred = ''

            #translations.append((corpus_indices[j], ' '.join(list_trans)))
            translations.append((corpus_indices[j], jth_pred))

        return translations






    # semi-parallelized version - computes beam in parallel
    # but still decodes 1 sequence at a time
    # termination condition - stop as soon as the current ts's
    # produced beam's most probable sequence ends in eos
    def getBeamSearchTranslation(self, decoder_inputs_batch, encoder_states, decoder_initial_state, src_lengths):
        sosIdx = self.sosIdx
        eosIdx = self.eosIdx
        bsz = encoder_states.size(0) # 1
        enc_d_hid = encoder_states.size(2) # varies depending on if bienc
        #decoder_inputs_tensor = torch.full((bsz, 1), sosIdx).long().cuda() # (bsz x 1)
        trg_lengths = torch.full((bsz,), 1).long().cuda()
        trg_lengths2 = torch.full((self.beam_size,), 1).long().cuda()
        mask = decoder_inputs_batch[0]
        corpus_indices = decoder_inputs_batch[1]
        max_src_len = src_lengths.max()    
        #translation = torch.tensor([]).long().cuda() # initialization for what will be (bsz x numTranslatedWordsSoFar)

        

        # -for first ts, the tracked variables have diff sizes than they will
        # for all other ts, bc for the first ts, there is only one previous seq
        # to extend (containing the sos token).

        # -for all other ts, there will be beam_size previous seqs, 
        # along with their total (summed) log probs, and the hidden states
        # that enabled their most recently added word's prediction

        # seqProbs init as zero vector of length 1 bc logProb of sentence beginning with sos is 0
        # (prob is 1)
        # at all other ts, will of length beam_size
        seqProbs = torch.zeros(1).cuda()

        # for first ts, extending a prev beam of size 1, so:
        # h_i and c_i will each be nl x 1 x hs, but
        # for all other ts, they will each be nl x beam_size x hs
        # seqs will be 1 x 1, but
        # for all other ts, will be beam_size x 1
        (h_i, c_i) = decoder_initial_state 
        seqs = torch.full((1, 1), sosIdx).long().cuda()
        input_i = self.trg_embeddings(seqs) # 1 x 1 x d_emb

        for i in range(max_src_len + self.decode_slack):
            #print(i)
            output, (h_i, c_i) = self.lstm(input_i, (h_i, c_i))
            
            if self.attention != None:
                # if i == 1:
                #     print("output: {}".format(output.size()))
                #     print("encoder_states: {}".format(encoder_states.size()))
                #     print("trg_lengths: {}".format(trg_lengths.size()))
                #     print("mask: {}".format(mask.size()))
                if i != 0:
                    # must line up the "batch" (more accurately, "beam") sizes
                    decoder_state = self.get_attStates(output, encoder_states.expand(self.beam_size, max_src_len, enc_d_hid), trg_lengths2, mask)
                else:
                    decoder_state = self.get_attStates(output, encoder_states, trg_lengths, mask)
                # get_attStates() returns (q x hs), which is (bsz x hs), in this case, so of correct shape

            # (bsz x hs) x (hs x V_trg) = (bsz x V_trg):
            preds = self.logsoftmax1(self.projectToV(decoder_state.data))

            # each beam gets to produce beam_size candidates
            topPreds, topIndices = torch.topk(preds.t(), self.beam_size, dim=0)
            topIndices = topIndices.view(-1)
            sum_seqProbs = (seqProbs.unsqueeze(0) + topPreds).view(-1)

            # keep the top beam_size of those candidates
            seqProbs, indices = torch.topk(sum_seqProbs, self.beam_size)
            prevIndices = indices % self.beam_size
            if i != 0: 
                h_i = h_i[:,prevIndices]
                c_i = c_i[:,prevIndices]
            else:
                # for first ts, the same hidden state produced all beam_size predictions
                h_i = h_i.expand(self.num_layers,self.beam_size,self.hidden_size).contiguous()
                c_i = c_i.expand(self.num_layers,self.beam_size,self.hidden_size).contiguous()
                seqs = seqs.expand(self.beam_size,1)
        
            words_to_append = topIndices[indices].unsqueeze(1)
            seqs = torch.cat((seqs[prevIndices], words_to_append), dim=1)

            # termination condition: 
            # most probable seq produced by the beam ends in eos
            if seqs[0][-1] == eosIdx:  
                #translation = seqs[0][1:].unsqueeze(0) # dn include the sos symbol
                break

            input_i = self.trg_embeddings(seqs[:,i+1].unsqueeze(1)) # beam_size x 1 x d_emb

        # translation is now (bsz x max_src_len+decode_slack)
        translation = seqs[0][1:].unsqueeze(0) # dn include the sos symbol
        tlist = translation.tolist()
        del translation
        translations = []
        for j in range(len(tlist)):
            try:
                eosIndex = tlist[j].index(eosIdx) # keep translation up to but not including eos
            except ValueError:
                eosIndex = len(tlist[j]) # never produced eos, so keep entire translation
            
            list_trans = [self.idx_to_trg_word[idx] for idx in tlist[j][:eosIndex]]
            try:
                #list_trans = naiveRecase(list_trans)
                #jth_pred = ' '.join(list_trans)
                jth_pred = formatPrediction(list_trans)

            except IndexError:
                print("predicted eos as first word")
                jth_pred = ''

            #translations.append((corpus_indices[j], ' '.join(list_trans)))
            translations.append((corpus_indices[j], jth_pred))

        return translations











    # returns the predicted translation as a string
    def unparallel_getBeamSearchTranslation(self, encoder_states, decoder_initial_state, src_lengths):
        # extract embedding for <sos>, and reshape as batch of 1 sequence of length 1, so can pass to lstm
        sosIdx = self.vocab['<sos>']
        eosIdx = self.vocab['<eos>']

        Lsrc = src_lengths[0] # there is only one src sentence, so extract its length
        (h_i, c_i) = decoder_initial_state # (nl x bsz x hs, nl x bsz x hs)
        trg_lengths = torch.tensor([1], device=self.dev).long()
        # seq is a list of 1D LongTensors, each of length 1
        beams = [(0.0, [torch.tensor([sosIdx], device=self.dev).long()], (h_i, c_i))] 
        mask = torch.zeros(1,1,Lsrc, device=self.dev) == 1 # byte tensor of all zeros, for if using attention

        # compute the decoder states one at a time:
        # (set some sanity check to cut off overly long translations)
        for i in range(Lsrc + 5): 
            beam_candidates = []
            for (curr_prob, seq, (h_i, c_i)) in beams:
                curr_wd_idx = seq[-1]
                if curr_wd_idx.item() == eosIdx:
                    beam_candidates.append((curr_prob, seq, (h_i, c_i)))
                    continue

                input_i = self.trg_embeddings(curr_wd_idx).view(1,1,-1)

                # input_i is 3D tensor of size (bsz x max_trg_len x input_size) = (1 x 1 x input_size)

                output, (h_i, c_i) = self.lstm(input_i, (h_i, c_i))
                # this lstm is unidirectional, so h_i is 3D tensor of shape:
                # (nl * nd x bsz x hs) = (nl x bsz x hs)
                # only want to project the output of the last layer:
                decoder_state = h_i[-1] # 2D tensor of size (bsz x hs) = (1 x hs)
                
                if self.attention != None:
                    # get_attStates() expects a 3D tensor of size (bsz x trg_max_len x hs) of decoder states,
                    # so reshape to accomodate hidden state for a single time step for a single sequence
                    # get_attStates() also expects a mask, so pass it (1 x 1 x Lsrc) mask of all 0s (merely leaves scores intact)
                    decoder_state = self.get_attStates(decoder_state.view(1,1,-1), encoder_states, trg_lengths, mask)
                    # get_attStates() returns (q x hs), which is (1 x hs), in this case, so of correct shape

                # (1 x hs) x (hs x V_trg) = (1 x V_trg):
                distOverNextWords = self.projectToV(decoder_state)
                # normalize into a probability distribution over possible next words:
                distOverNextWords = self.logsoftmax1(distOverNextWords).squeeze() # 1D of length |V|
                
                #??replace with torch.topk??
                _, top_indices = torch.sort(-distOverNextWords) # sort in descending order (log domain)
                top_b_wdIdxs = top_indices[:self.beam_size] 
                
                for idx in top_b_wdIdxs: # idx is 1D tensor consisting of a single element (a word idx)

                    ###!!!changed
                    #seqProb = curr_prob + distOverNextWords[idx]
                    lp = ((5 + len(seq)) ** .6) / (6 ** .6)
                    seqProb = curr_prob + (distOverNextWords[idx] / lp) # seqProb is 1D tensor of length 1 (add scalar to 1D tensor of length 1)
                    subtrans = seq + [idx] # subtrans holds in last position the idx whose emb will be inputted at next ts
                    # (h_i, c_i) is the resulting state after inputting the emb of the word for this beam
                    # (the word at last position of seq). it is the initial state for when inputting idx at next ts
                    beam_candidates.append((seqProb, subtrans, (h_i, c_i)))

            if len(beam_candidates) == self.beam_size and not i == 0:
                # no new beam candidates were produced, bc each already ends in <eos>
                # (exception, on first iter, there will always be 5, bc produced 5 candidates from single sos,
                # so kept all of them)
                beams = beam_candidates
                break
            # of the at most b*b candidates that were generated this iteration, keep the top b
            beams = sorted(beam_candidates, key = lambda b: b[0], reverse = True)[:self.beam_size]

        translation = beams[0][1] # list of wd idx's of the most probable sentence in beams
        translation = [self.idx_to_trg_word[idx.item()] for idx in translation][1:] # sentence as list of string forms of words
        # (dn include sos token)

        # real data does not append the <eos> token
        return ' '.join(translation[:-1]) 
        










    # returns concatenation of the queries with their context vectors (bsz x (2*hidden_size))
    # padded_decoder_states is bsz x max_trg_len x decoder_hs
    # padded_encoder_states is bsz x max_src_len x encoder_hs
    # src_lengths is 1D of length bsz
    # trg_lengths is 1D of length bsz
    # mask is bsz x max_trg_len x max_src_len
    def get_attStates(self, padded_decoder_states, padded_encoder_states, trg_lengths, mask):
        
        # if testing, then trg_lengths will always be entirely 1's,
        # and mask will be (bsz x 1 x max_src_len)
        scores = torch.bmm(padded_decoder_states, padded_encoder_states.transpose(1,2)) # (bsz x max_trg_len x max_src_len)
        scores = scores / self.sqrt_hs # scale by sqrt of hs (no-op if do not use scaling)
        scores = scores.masked_fill(mask, -float('inf'))
        scores = self.softmax2(scores)
        padded_contexts = torch.bmm(scores, padded_encoder_states) # bsz x max_trg_len x encoder_hs
        #packed_contexts, _ = pack_padded_sequence(padded_contexts, trg_lengths, batch_first=True)
        del scores
        #packed_contexts = pack_padded_sequence(padded_contexts, trg_lengths, batch_first=True)


        #del padded_contexts
        #packed_contexts = packed_contexts.data

        #packed_decoder_states, _ = pack_padded_sequence(padded_decoder_states, trg_lengths, batch_first=True) 
        #packed_decoder_states = pack_padded_sequence(padded_decoder_states, trg_lengths, batch_first=True) 
        #packed_decoder_states = packed_decoder_states.data

        # concatenate contexts with the original decoder states
        #concated = torch.cat((packed_contexts, packed_decoder_states), dim=1) # 2D tensor of shape (q x 2*hs)
        
        
        concated = torch.cat((padded_contexts, padded_decoder_states), dim=2) # 3D tensor of shape (bsz x max_trg_len x 2*hs)

        # apply var inf before and after
        #self.myVarInf(concated, p=???)



        #del packed_contexts, packed_decoder_states
        #attStates = torch.cat((packed_contexts, packed_decoder_states), dim=1) # 2D tensor of shape (q x 2*hs)
        #lintrans = self.projectBackToEmbSize(concated) # (q x input_size)
        attStates = self.tanh(self.att_layer(concated)) # (bsz x max_trg_len x hidden_size)
        
        #attStates = self.myVarInf(attStates, self.att_drop)
        attStates = pack_padded_sequence(attStates, trg_lengths, batch_first=True) # (q x hidden_size)
        ### this is what Luong did, but could also decide to just directly project this concated vector to V
        # apply tanh to generate attentional hidden state for current time step (implement later)
        
        #del concated
        #attStates = self.tanh(lintrans) # (q x input_size)
        #del lintrans
        #!!!new: apply var inf to attentional output


        return attStates














def beam_search_translate(self, decoder_inputs, encoder_states, initial_state):
        # -for first ts, the tracked variables have diff sizes than they will
        # for all other ts, bc for the first ts, there is only one previous seq
        # to extend (containing the sos token).

        # -for all other ts, there will be beam_size previous seqs, 
        # along with their total (summed) log probs, and the hidden states
        # that enabled their most recently added word's prediction

        # seqProbs init as zero vector of length 1 bc logProb of sentence beginning with sos is 0
        # (prob is 1)
        # at all other ts, will of length beam_size
        beam_likelihoods = torch.zeros(1).cuda() 

        # for first ts, extending a prev beam of size 1, so:
        # h_i and c_i will each be nl x 1 x hs, but
        # for all other ts, they will each be nl x beam_size x hs
        # seqs will be 1 x 1, but
        # for all other ts, will be beam_size x 1
        (h_i, c_i) = initial_state 
        beam_sequences = torch.full((1, 1), int(self.sos_idx)).long().cuda()
        input_i = self.embed(beam_sequences) # 1 x 1 x d_emb

        ### NOTE: unlike training step, need to normalize the predictions with log softmax, bc need to track the running log probabilities of the sequences
        for i in range(decoder_inputs["max_src_len"] + self.decode_slack):
            if i != 0:
                dists_i, (h_i, c_i) = self.decode_step(input_i, (h_i, c_i), decoder_inputs["mask"], encoder_states.expand(self.beam_size, -1, -1), decoder_inputs["mask"])
            else:
                dists_i, (h_i, c_i) = self.decode_step(input_i, (h_i, c_i), decoder_inputs["mask"], encoder_states, decoder_inputs["mask"])


            # each beam gets to produce beam_size candidates
            top_preds, top_words = torch.topk(dists_i.t(), self.beam_size, dim=0)
            top_words = top_words.view(-1)
            summed_likelihoods = (beam_likelihoods.unsqueeze(0) + top_preds).view(-1)

            # keep the top beam_size of those candidates
            top_summed_likelihoods, top_summed_likelihood_indices = torch.topk(summed_likelihoods, self.beam_size)
            beam_sequence_indices = top_summed_likelihood_indices % self.beam_size
            if i != 0: 
                h_i = h_i[:,beam_sequence_indices]
                c_i = c_i[:,beam_sequence_indices]
            else:
                # for first ts, the same hidden state produced all beam_size predictions
                h_i = h_i.expand(self.num_layers,self.beam_size,self.hidden_size).contiguous()
                c_i = c_i.expand(self.num_layers,self.beam_size,self.hidden_size).contiguous()
                sequences = sequences.expand(self.beam_size,1)
        
            # sequence extensions. the indices of the words to extend each sequence of beam with
            sequence_extensions = top_words[top_summed_likelihood_indices].unsqueeze(1)
            beam_sequences = torch.cat((beam_sequences[beam_sequence_indices], sequence_extensions), dim=1)

            # termination condition: 
            # most probable seq produced by the beam ends in eos
            if beam_sequences[0][-1] == self.eos_idx:  
                break

            input_i = self.embed(beam_sequences[:,i+1].unsqueeze(1)) # beam_size x 1 x d_emb

        # translation is now (bsz x max_src_len+decode_slack)
        translation = beam_sequences[0][1:].unsqueeze(0) # dn include the sos symbol

        return translation

