import torch.nn as nn
import torch
import numpy as np

# Global variables for configuration
encoder_vocab_length = 10000  # TODO: Temp vocab data length
decoder_vocab_length = 10000  # TODO: Temp vocab data length
encoder_sequence_length = 256
decoder_sequence_length = 256
layer_length = 6
head_length = 4
head_depth = 64
hidden_depth = head_length * head_depth  # 256
feed_forward_depth = hidden_depth * 4  # 1024


def get_sinusoid_table(sequence_length):
	def calculate_angle(position, i):
		return position / np.power(10000, 2 * (i // 2) / hidden_depth)

	def get_position_angle_vector(position):
		return [calculate_angle(position, i) for i in range(hidden_depth)]

	sinusoid_table = np.array([get_position_angle_vector(i) for i in range(sequence_length)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # put sine values at even indexes
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # put cosine values at odd indexes
	return torch.from_numpy(sinusoid_table)


def get_attention_pad_mask(q, k):
	batch_size, q_length = q.size()
	batch_size, k_length = k.size()
	attention_pad_mask = k.data.eq(0).unsqeeze(1).expand(batch_size, q_length, k_length)
	return attention_pad_mask


def get_attention_decoder_mask(inputs):
	pad_mask = get_attention_pad_mask(inputs, inputs)
	decoder_mask = torch.ones_like(inputs).unsqueeze(-1).expand(inputs.size(0), inputs.size(1), inputs.size(1)).triu(1)
	decoder_self_attention_mask = torch.gt((pad_mask + decoder_mask), 0)
	return decoder_self_attention_mask


class ScaledDotProductAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.dropout = nn.Dropout(0.1)
		self.scale = 1 / (head_length ** 0.5)

	def forward(self, q, k, v, mask):
		# MatMul
		outputs = torch.matmul(q, k.transpose(-1, -2))

		# Scale
		outputs = torch.mul(outputs, self.scale)

		# Mask
		outputs.masked_fill_(mask, -1e9)

		# SoftMax
		outputs = nn.Softmax(dim=-1)(outputs)
		outputs = self.dropout(outputs)

		# MatMul
		context = torch.matmul(outputs, v)
		return context, outputs


class MultiHeadAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.q_w = nn.Linear(hidden_depth, hidden_depth)
		self.k_w = nn.Linear(hidden_depth, hidden_depth)
		self.v_w = nn.Linear(hidden_depth, hidden_depth)
		self.scaled_dot_product_attention = ScaledDotProductAttention()
		self.linear = nn.Linear(hidden_depth, hidden_depth)
		self.dropout = nn.Dropout(0.1)

	def forward(self, q, k, v, mask):
		# Linear
		batch_size = q.size(0)
		q_sequence = self.q_w(q).view(batch_size, -1, head_length, head_depth).transpose(1, 2)
		k_sequence = self.k_w(k).view(batch_size, -1, head_length, head_depth).transpose(1, 2)
		v_sequence = self.v_w(v).view(batch_size, -1, head_length, head_depth).transpose(1, 2)
		mask = mask.unsqeeze(1).repeat(1, head_length, 1, 1)

		# Attention
		context, attention_probability = self.scaled_dot_product_attention(q_sequence, k_sequence, v_sequence, mask)

		# Concat
		context = context.transpose(1, 2).contigous().view(batch_size, -1, hidden_depth)

		# Linear
		output = self.linear(context)

		# Dropout
		output = self.dropout(output)

		return output, attention_probability


class PositionWiseFeedForwardNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(hidden_depth, feed_forward_depth, 1)
		self.conv2 = nn.Conv1d(feed_forward_depth, hidden_depth, 1)
		self.activate = nn.functional.relu  # Other activations like GELU has better performance, but the paper uses ReLU
		self.dropout = nn.Dropout(0.1)

	def forward(self, inputs):
		# Conv
		output = self.conv1(inputs.transpose(1, 2))

		# Activate
		output = self.activate(output)

		# Conv
		output = self.conv2(output)

		# Dropout
		output = self.dropout(output)
		return output


class EncoderLayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.attention = MultiHeadAttention()
		self.norm1 = nn.LayerNorm(hidden_depth, eps=1e-6)
		self.feed_forward = PositionWiseFeedForwardNetwork()
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
		self.input_embedding = nn.Embedding(encoder_vocab_length, hidden_depth)
		sinusoid_table = get_sinusoid_table(encoder_sequence_length + 1)
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

		for layer in self.layers:
			outputs = layer(outputs, attention_mask)  # The output of the layer is new input.

		return outputs


class DecoderLayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.self_attention = MultiHeadAttention()
		self.norm1 = nn.LayerNorm(hidden_depth, eps=1e-6)
		self.encoder_attention = MultiHeadAttention()
		self.norm2 = nn.LayerNorm(hidden_depth, eps=1e-6)
		self.feed_forward = PositionWiseFeedForwardNetwork()
		self.norm3 = nn.LayerNorm(hidden_depth, eps=1e-6)

	def forward(self, inputs, encoder_outputs, self_attention_mask, encoder_attention_mask):
		# Masked Multi-Head Attention
		self_attention_outputs = self.self_attention(inputs, inputs, inputs, self_attention_mask)

		# Add inputs & Norm
		self_attention_outputs = self.norm1(inputs + self_attention_outputs)

		# Multi-Head Attention with Key and Value of encoder
		encoder_attention_outputs = \
			self.encoder_attention(self_attention_outputs, encoder_outputs, encoder_outputs, encoder_attention_mask)

		# Add inputs & Norm
		encoder_attention_outputs = self.norm2(self_attention_outputs + encoder_attention_outputs)

		# Feed Forward
		feed_forward_outputs = self.feed_forward(encoder_attention_outputs)

		# Add inputs & Norm
		feed_forward_outputs = self.norm3(encoder_attention_outputs + feed_forward_outputs)

		return feed_forward_outputs


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.decoder_embedding = nn.Embedding(decoder_vocab_length, hidden_depth)
		sinusoid_table = get_sinusoid_table(decoder_sequence_length + 1)
		self.position_encoding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
		self.layers = nn.ModuleList([DecoderLayer() for _ in range(layer_length)])

	def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
		# Masking for Positional Encoding
		positions = torch.arange(decoder_inputs.size(1), device=decoder_inputs.device, dtype=decoder_inputs.dtype) \
			            .expand(decoder_inputs.size(0), decoder_inputs.size(1)).contiguous() + 1
		position_mask = decoder_inputs.eq(0)
		positions.masked_fill_(position_mask, 0)

		# Decoder Layer loop n times
		decoder_outputs = self.decoder_embedding(decoder_inputs) + self.position_encoding(positions)
		self_attention_mask = get_attention_decoder_mask(decoder_inputs)
		encoder_attention_mask = get_attention_pad_mask(decoder_inputs, encoder_inputs)

		for layer in self.layers:
			decoder_outputs = layer(decoder_outputs, encoder_outputs, self_attention_mask, encoder_attention_mask)

		return decoder_outputs


class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, encoder_inputs, decoder_inputs):
		encoder_outputs = self.encoder(encoder_inputs)
		decoder_outputs = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
		# Linear
		# SoftMax
		return decoder_outputs
