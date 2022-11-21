import math
import torch
import torch.nn as nn

def get_extended_attention_mask(attention_mask, autoregressive=False):

    dtype = torch.float16

    extended_attention_mask = attention_mask[:, None, None, :]

    if autoregressive is True:

      subsequent_mask = torch.ones_like(extended_attention_mask).expand(-1, -1, attention_mask.size(1), -1)
      subsequent_mask = subsequent_mask.triu(diagonal=1)
      subsequent_mask = torch.lt(subsequent_mask,1)

      extended_attention_mask = torch.gt((extended_attention_mask+subsequent_mask), 1).int()

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.feed_forward_dim, bias=False),
                                          nn.ReLU(),
                                          nn.Dropout(config.drop_out_raito),
                                          nn.Linear(config.feed_forward_dim, config.d_model, bias=False),
                                          nn.Dropout(config.drop_out_raito))

    def forward(self, inputs):
        return self.feed_forward(inputs)
    
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states

class T5Encoder(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.word_embedding = embedding
        self.layers = nn.ModuleList(
            [T5EncoderLayer(config) for i in range(config.num_enc_layers)]
        )

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.drop_out_raito)


    def forward(self,
                input_ids,
                attention_mask):
      
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()

        self_attention_mask = get_extended_attention_mask(attention_mask)

        outputs = self.word_embedding(input_ids)
      
        self_attn_probs = []
        for i, layer in enumerate(self.layers):
            outputs, self_attn_prob = layer(outputs,
                                            self_attention_mask,
                                            )
            self_attn_probs.append(self_attn_prob)

        outputs = self.final_layer_norm(outputs)
        outputs = self.dropout(outputs)

        return outputs, self_attn_probs, self_attention_mask  

class T5EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attention = T5Attention(config, has_relative_attention_bias=True, autoregressive=False)

        self.feed_forward_norm= T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feed_forward  = PoswiseFeedForward(config)

    def forward(self, 
                inputs, 
                self_attention_mask):

        outputs, self_attn_prob = self.self_attention(query=self.self_attention_norm(inputs), 
                                                      key=None, 
                                                      value=None, 
                                                      attention_mask=self_attention_mask,
                                                      )        
        outputs = inputs + outputs

        inputs = outputs
        outputs = self.feed_forward(self.feed_forward_norm(inputs))
        outputs = inputs + outputs

        # clamp inf values to enable fp16 training
        if outputs.dtype == torch.float16 and torch.isinf(outputs).any():
            clamp_value = torch.finfo(outputs.dtype).max - 1000
            outputs = torch.clamp(outputs, min=-clamp_value, max=clamp_value)

        return outputs, self_attn_prob

class T5Attention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=True, autoregressive=False):
        super().__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        self.d_head = int(self.d_model / self.num_att_heads)
        self.scale = self.d_head ** 0.5
        self.has_relative_attention_bias = has_relative_attention_bias
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head, bias=False)
        self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_num_buckets = config.relative_attention_num_buckets
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, config.num_att_heads)
            self.bidirectional = not autoregressive

        self.dropout = nn.Dropout(config.drop_out_raito)

        self.fc = nn.Linear(self.d_head * self.num_att_heads, self.d_model, bias=False)

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets     
            
    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        global relative_position
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position=relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values      


    def forward(self,
                query,
                key=None,
                value=None,
                attention_mask=None,
                ):
      
        if key is None and value is None:
            key = value = query
      
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        scores = torch.matmul(query, key.transpose(-2, -1))        
        
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.num_att_heads, query_len, key_len), device=scores.device, dtype=scores.dtype
            )
        else:
            position_bias = self.compute_bias(query_len, key_len)

        position_bias = position_bias + attention_mask
        
        scores = scores + position_bias
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)

        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        context = self.fc(context)
        context = self.dropout(context)

        return context, attn_prob

class T5Decoder(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config=config
        
        self.word_embedding = embedding
        self.layers = nn.ModuleList(
            [T5DecoderLayer(config) for i in range(config.num_enc_layers)]
        )

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.drop_out_raito)

        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self,
                input_ids,
                attention_mask=None,
                enc_outputs=None,
                enc_attention_mask=None):
      
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()

        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=True)

        outputs = self.word_embedding(input_ids)
      
        self_attn_probs, cross_attn_probs = [], []
        for i, layer in enumerate(self.layers):
            outputs, self_attn_prob, cross_attn_prob = layer(outputs,
                                                             self_attention_mask,
                                                             enc_outputs,
                                                             enc_attention_mask,
                                                             )
            self_attn_probs.append(self_attn_prob)
            cross_attn_probs.append(cross_attn_prob)   

        outputs = self.final_layer_norm(outputs)
        outputs = self.dropout(outputs)

        outputs = self.fc(outputs)

        return outputs, self_attn_probs, cross_attn_probs

class T5DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attention_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attention = T5Attention(config, has_relative_attention_bias=True, autoregressive=True)
        
        self.cross_attention_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.cross_attention = T5Attention(config, has_relative_attention_bias=False, autoregressive=False)

        self.feed_forward_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feed_forward = PoswiseFeedForward(config)

    def forward(self,
                inputs,
                self_attention_mask,
                enc_outputs,
                cross_attention_mask,
                ):

        outputs, self_attn_prob = self.self_attention(query=self.self_attention_norm(inputs), 
                                                      key=None, 
                                                      value=None, 
                                                      attention_mask=self_attention_mask,
                                                      )
        outputs = inputs + outputs
        
        # clamp inf values to enable fp16 training
        if outputs.dtype == torch.float16 and torch.isinf(outputs).any():
            clamp_value = torch.finfo(outputs.dtype).max - 1000
            outputs = torch.clamp(outputs, min=-clamp_value, max=clamp_value)

        inputs = outputs
        outputs, cross_attn_prob = self.cross_attention(query=self.cross_attention_norm(inputs), 
                                                        key=enc_outputs, 
                                                        value=enc_outputs, 
                                                        attention_mask=cross_attention_mask,
                                                        )
        outputs = inputs + outputs

        if outputs.dtype == torch.float16 and torch.isinf(outputs).any():
            clamp_value = torch.finfo(outputs.dtype).max - 1000
            outputs = torch.clamp(outputs, min=-clamp_value, max=clamp_value)

        inputs = outputs
        outputs = self.feed_forward(inputs)
        outputs = inputs + outputs

        return outputs, self_attn_prob, cross_attn_prob 

class T5_Model(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.config=config

      if config.share_embedding is True:
          self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
          self.encoder = T5Encoder(config, self.shared_embedding)
          self.decoder = T5Decoder(config, self.shared_embedding)
      
      else:
          self.encoder_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
          self.decoder_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
  
          self.encoder = T5Encoder(config, self.encoder_embedding)
          self.decoder = T5Decoder(config, self.decoder_embedding)

      self.init_weights()

    def init_weights(self):
        # Initialize weights for each layer
        self.apply(self.init_layer_weights)

    # ref huggingface
    # https://huggingface.co/transformers/v4.9.2/_modules/transformers/models/electra/modeling_electra.html#ElectraPreTrainedModel
    def init_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            module.eps = self.config.layer_norm_eps

    def forward(self,
                enc_input_ids,
                enc_attention_mask=None,
                dec_input_ids=None,
                dec_attention_mask=None,
                ):

        enc_outputs, enc_self_attn_probs, enc_attention_mask = self.encoder(enc_input_ids,
                                                                            enc_attention_mask,
                                                                            )
        
        dec_outputs, dec_self_attn_probs, dec_cross_attn_probs = self.decoder(input_ids=dec_input_ids,
                                                                              attention_mask=dec_attention_mask,
                                                                              enc_outputs=enc_outputs,
                                                                              enc_attention_mask=enc_attention_mask)

        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_cross_attn_probs
