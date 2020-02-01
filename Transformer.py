import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, inputs):
		pass


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, inputs, ):
		pass


class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, encoder_inputs, decoder_inputs):
		encoder_self_attention_probabilities = self.encoder(encoder_inputs)
		decoder_outputs = self.decoder(encoder_self_attention_probabilities, decoder_inputs)
		return decoder_outputs
