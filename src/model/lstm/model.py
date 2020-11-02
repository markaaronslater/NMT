import torch.nn as nn

from src.model.lstm.encoder import Encoder
from src.model.lstm.decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, hyperparams):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(hyperparams)
        self.decoder = Decoder(hyperparams)

        if hyperparams["vocab_type"] == 'subword_joint': 
            # learn single vocab for encoder and decoder
            self.encoder.embed.weight = self.decoder.embed.weight


    def forward(self, encoder_inputs, decoder_inputs):
        encoder_states, decoder_initial_state = self.encoder(encoder_inputs)
        return self.decoder(decoder_inputs, encoder_states, decoder_initial_state)