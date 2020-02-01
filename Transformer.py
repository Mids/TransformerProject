import torch.nn as nn
import torch
import numpy as np

# Global variables for configuration
input_vocab_length = 10000  # TODO: Temp vocab data length
encoder_sequence_length = 256
decoder_sequence_length = 256
hidden_depth = 256
layer_length = 6


def get_sinusoid_table(sequence_length):
	def cal_angle(position, i):
		return position / np.power(10000, 2 * (i // 2) / hidden_depth)

	def get_position_angle_vector(position):
		return [cal_angle(position, i) for i in range(hidden_depth)]

	sinusoid_table = np.array([get_position_angle_vector(i) for i in range(sequence_length)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # put sine values at even indexes
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # put cosine values at odd indexes
	return torch.from_numpy(sinusoid_table)


def get_attention_pad_mask(q, k):
	batch_size, q_length = q.size()
	batch_size, k_length = k.size()
	attention_pad_mask = k.data.eq(0).unsqeeze(1).expand(batch_size, q_length, k_length)
	return attention_pad_mask


class MultiHeadAttention(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, q, k, v, mask):
		pass


class FeedForward(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, inputs):
		pass


class EncoderLayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention = MultiHeadAttention()
		self.norm1 = nn.LayerNorm(hidden_depth, eps=1e-6)
		self.feed_forward = FeedForward()
		self.norm2 = nn.LayerNorm(hidden_depth, eps=1e-6)

	def forward(self, inputs, mask):
		# Multi-Head Attention
		attention_outputs, attention_probability = self.attention(inputs, inputs, inputs, mask)  # Q == K == V

		# Add inputs & Norm
		attention_outputs = self.norm1(inputs + attention_outputs)

		# Feed Forward
		feed_forward_outputs = self.feed_forward(attention_outputs)

		# Add inputs & Norm
		feed_forward_outputs = self.norm2(attention_outputs + feed_forward_outputs)

		return feed_forward_outputs


class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.input_embedding = nn.Embedding(input_vocab_length, hidden_depth)
		sinusoid_table = get_sinusoid_table(encoder_sequence_length)
		self.position_encoding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
		self.layers = nn.ModuleList([EncoderLayer() for _ in range(layer_length)])

	def forward(self, inputs):
		# Masking for Positional Encoding
		positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype) \
			            .expand(inputs.size(0), inputs.size(1)).contiguous() + 1  # +1 to start from 1, not 0
		position_mask = inputs.eq(0)
		positions.masked_fill_(position_mask, 0)

		# Encoder Layer loop n times
		outputs = self.input_embedding(inputs) + self.position_encoding(positions)
		attention_mask = get_attention_pad_mask(inputs, inputs)  # Q, K, and V is identical in encoder
		attention_probabilities = []

		for layer in self.layers:
			outputs, probability = layer(outputs, attention_mask)  # The output of the layer is new input.
			attention_probabilities.append(probability)

		return attention_probabilities


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, inputs, encoder_self_attention_probabilities):
		# Output Embedding
		# Positional Encoding
		# Decoder Layer loop n times
		# Linear
		# Softmax
		pass


class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, encoder_inputs, decoder_inputs):
		encoder_self_attention_probabilities = self.encoder(encoder_inputs)
		decoder_outputs = self.decoder(decoder_inputs, encoder_self_attention_probabilities)
		return decoder_outputs
