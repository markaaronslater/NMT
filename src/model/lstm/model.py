import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, hyperparams, vocab_sizes):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hyperparams, vocab_sizes["encoder"])
        self.decoder = Decoder(hyperparams, vocab_sizes["decoder"])

        if self.encoder.bidirectional and not self.encoder.project:
            assert self.decoder.hidden_size == 2 * self.encoder.hidden_size
        else:    
            assert self.decoder.hidden_size == self.encoder.hidden_size

        if self.encoder.decoder_init_scheme == "layer_to_layer":
            assert self.decoder.num_layers == self.encoder.num_layers

        if hyperparams["vocab_type"] == 'subword_joint': 
            # learn single vocab for encoder and decoder
            self.encoder.embed.weight = self.decoder.embed.weight


    def forward(self, encoder_inputs, decoder_inputs):
        encoder_states, decoder_initial_state = self.encoder(encoder_inputs)
        return self.decoder(decoder_inputs, encoder_states, decoder_initial_state)



