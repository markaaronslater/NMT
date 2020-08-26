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

       
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bi_enc, dropout=self.dropout, batch_first=True)
        #self.init_lstm_dc()
        
        
        if self.bi_enc:
            self.bridge = nn.Linear(2*self.hidden_size, self.hidden_size)
            self.projKeys = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.src_embeddings = nn.Embedding(self.vocab_size, self.input_size)
        self.src_embeddings.weight.data.uniform_(-.1, .1)
        self.out_drop = encoder_params['out_drop']
        self.tanh = nn.Tanh()

    


      







    def forward(self, encoder_inputs_batch):
        encoder_inputs_tensor = encoder_inputs_batch[0]
        src_lengths = encoder_inputs_batch[1]
        idxs_in_sorted_enc_inputs_tensor = encoder_inputs_batch[2]

        srcEmbeddings = self.src_embeddings(encoder_inputs_tensor)
        
        packed_input = pack_padded_sequence(srcEmbeddings, src_lengths, batch_first=True)
        

        packed_output, (hn, cn) = self.lstm(packed_input)
        encoder_states, _ = pad_packed_sequence(packed_output, batch_first=True) # 3D tensor of size (bsz x max_len x 2*enc_hs)
        del packed_output

        bsz = encoder_states.size(0) # varies for "remainder" batch
        d_hid = encoder_states.size(2) # varies if bidirectional
        #out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, d_hid).cuda(), p=self.out_drop, training=self.training)
        #encoder_states = encoder_states * out_mask


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

        if self.init_scheme == "layer_to_layer":
            return encoder_final

        elif self.init_scheme == "final_to_first":
            decoder_initial_h = torch.zeros(self.num_layers, bsz, self.hidden_size, device=self.dev)
            decoder_initial_c = torch.zeros(self.num_layers, bsz, self.hidden_size, device=self.dev)
            # initialize first layer of decoder with last layer of encoder
            decoder_initial_h[0] = encoder_final[0][-1] 
            decoder_initial_c[0] = encoder_final[1][-1] 


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

        
        self.out_drop = decoder_params['out_drop']
        
        self.customLSTM = decoder_params['customLSTM']
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True)
        

        
        

        self.trg_embeddings = nn.Embedding(self.vocab_size, self.input_size)

        

        self.softmax1 = nn.Softmax(dim=1) 
        self.softmax2 = nn.Softmax(dim=2) 
        self.logsoftmax1 = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

        #new!!! TIE THE WEIGHTS. decoder_hidden must equal decoder input_size
        if decoder_params['tie_weights']:
            self.projectToV = nn.Linear(self.input_size, self.vocab_size) 
            #assert self.hidden_size == self.input_size

            ###???how do these dimensions work out, again???
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
        


    

    def forward(self, decoder_inputs_batch, padded_encoder_states, decoder_initial_state, src_lengths):
        if not self.tf: # not teaching forcing (making prediction for a single sentence instead)
            ###???isnt it predicting entire dev batch, not a single sentence???
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

            trgEmbeddings = self.trg_embeddings(decoder_inputs_tensor) # 3D tensor of shape (bsz x max_len x input_size)
            
            packed_input = pack_padded_sequence(trgEmbeddings, trg_lengths, batch_first=True)

            # packed_input is a tuple, whose first comp is the packed sequence, and second comp is "batchsizes",
            # which holds, at position i, the number of inputs for time step i 
            return self.computeProbDists(packed_input, padded_encoder_states, decoder_initial_state, src_lengths, trg_lengths, mask)
        

    # model makes its predictions for next words given the actual previous words (teacher-forcing)
    def computeProbDists(self, packed_input, padded_encoder_states, decoder_initial_state, src_lengths, trg_lengths, mask):
        

        packed_output, _ = self.lstm(packed_input, decoder_initial_state)
        del packed_input


        # Let q be total length of all trg seqs of the batch. then packed_decoder_states is 2D tensor of shape (q x hs)
        if self.attention != None:
            # convert states to attentional states before projecting
            decoder_states, _ = pad_packed_sequence(packed_output, batch_first=True)
            del packed_output
            bsz = decoder_states.size(0)


            #out_mask = torch.nn.functional.dropout(torch.ones(bsz, 1, self.hidden_size).cuda(), p=self.out_drop, training=self.training)
            #decoder_states = decoder_states * out_mask

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
        decoder_inputs_tensor = torch.full((bsz, 1), sosIdx, dtype=torch.long).cuda() # (bsz x 1)
        trg_lengths = torch.full((bsz,), 1, dtype=torch.long).cuda()

        mask = decoder_inputs_batch[0]
        corpus_indices = decoder_inputs_batch[1]
        max_src_len = src_lengths.max()    
        translation = torch.tensor([]).long().cuda() # initialization for what will be (bsz x numTranslatedWordsSoFar)

        
        
        input_i = self.trg_embeddings(decoder_inputs_tensor) # (bsz x 1 x input_size)

        max_src_len = src_lengths.max()
        (h_i, c_i) = decoder_initial_state
        
        # compute the decoder states one at a time:
        # (set some sanity check to cut off overly long translations)
        for i in range(max_src_len + self.decode_slack): 
            output, (h_i, c_i) = self.lstm(input_i, (h_i, c_i))
            
            # this lstm is unidirectional, so h_i is 3D tensor of shape:
            # (nl * nd x bsz x hs) = (nl x bsz x hs)

            if self.attention != None:
                # get_attStates() expects a 3D tensor of size (bsz x trg_max_len x hs) of decoder states,
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
                jth_pred = formatPrediction(list_trans)

            except IndexError:
                print("predicted eos as first word")
                jth_pred = ''

            translations.append((corpus_indices[j], jth_pred))

        return translations






    # semi-parallelized version - computes beam in parallel
    # but still decodes 1 sequence at a time
    # termination condition - stop as soon as the current ts's
    # produced beam's most probable sequence ends in eos

    ### NOTE: see experimental_encoderdecoder.py for unparallelized version
    def getBeamSearchTranslation(self, decoder_inputs_batch, encoder_states, decoder_initial_state, src_lengths):
        sosIdx = self.sosIdx
        eosIdx = self.eosIdx
        bsz = encoder_states.size(0) # 1
        enc_d_hid = encoder_states.size(2) # varies depending on if bienc


        ### fix these full specs!!!
        trg_lengths = torch.full((bsz,), 1).long().cuda()
        trg_lengths2 = torch.full((self.beam_size,), 1).long().cuda()
        mask = decoder_inputs_batch[0]
        corpus_indices = decoder_inputs_batch[1]
        max_src_len = src_lengths.max()    
        

        

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
                jth_pred = formatPrediction(list_trans)

            except IndexError:
                print("predicted eos as first word")
                jth_pred = ''

            translations.append((corpus_indices[j], jth_pred))

        return translations











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
        del scores

        # concatenate contexts with the original decoder states
        concated = torch.cat((padded_contexts, padded_decoder_states), dim=2) # 3D tensor of shape (bsz x max_trg_len x 2*hs)

         # 2D tensor of shape (q x 2*hs)
        attStates = self.tanh(self.att_layer(concated)) # (bsz x max_trg_len x hidden_size)
        
        attStates = pack_padded_sequence(attStates, trg_lengths, batch_first=True) # (q x hidden_size)
        

        return attStates








