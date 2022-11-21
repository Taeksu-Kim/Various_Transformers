import math
import torch
import torch.nn as nn

def PositionalEncoding(max_seq_len, d_model):
    '''
    PE_(pos, 2i)   =  sin(pos / power(10000, 2i / d_model))
    PE_(pos, 2i+1) =  cos(pos / power(10000, 2i / d_model))
    '''
    pe = torch.zeros([max_seq_len, d_model])
    position = torch.arange(max_seq_len).unsqueeze(1).repeat(1, d_model) # pos, [seq_len, d_model]
    div_value = torch.pow(10000, torch.arange(0, d_model, 2) / d_model) # power(10000, 2i / d_model)
    pe[:, 0::2] = torch.sin(position[:, 0::2] / div_value) # sin for 2i
    pe[:, 1::2] = torch.cos(position[:, 1::2] / div_value) # cos for 2i+1
    pe = pe.unsqueeze(0) # [bs(1), seq_len, d_model]
    
    return pe

def get_extended_attention_mask(attention_mask, autoregressive=False):

    dtype = torch.float16

    extended_attention_mask = attention_mask[:, None, None, :]

    if autoregressive is True:

      subsequent_mask = torch.ones_like(extended_attention_mask, device=attention_mask.device).expand(-1, -1, attention_mask.size(1), -1)
      subsequent_mask = subsequent_mask.triu(diagonal=1)
      subsequent_mask = torch.lt(subsequent_mask,1)

      extended_attention_mask = torch.gt((extended_attention_mask+subsequent_mask), 1).int()

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        
        self.d_head = int(self.d_model / self.num_att_heads)
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        self.scaled_dot_attn = ScaledDotProductAttention(config, self.d_head)
        self.fc = nn.Linear(self.d_head * self.num_att_heads, self.d_model)

    def forward(self, query, key, value, attention_mask):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        context, attn_prob = self.scaled_dot_attn(query, key, value, attention_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        output = self.fc(context)
        
        return output, attn_prob

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.scale = d_head ** 0.5

    def forward(self, query, key, value, attention_mask):

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [bs, num_heads, query_len, key_len]
        
        scores = scores + attention_mask
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
                                                  
        return context, attn_prob

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForward, self).__init__()      

        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.feed_forward_dim),
                                          nn.Dropout(config.drop_out_raito),
                                          nn.ReLU(),
                                          nn.Linear(config.feed_forward_dim, config.d_model),
                                          nn.Dropout(config.drop_out_raito))

    def forward(self, inputs):
        return self.feed_forward(inputs)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.sqrt_dim = math.sqrt(config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_enc_len, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_enc_layers)]
        )

    def forward(
        self, 
        input_ids, 
        attention_mask=None,
        ):

        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()

        word_embeds = self.word_embedding(input_ids) * self.sqrt_dim
        position_embeds = self.pos_encoding[:, :seq_len].to(input_ids.device)

        outputs = word_embeds + position_embeds

        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=False)    

        self_attn_probs = []
        for layer in self.layers:
            outputs, self_attn_prob = layer(outputs, 
                                            self_attention_mask,
                                            )
            self_attn_probs.append(self_attn_prob)
        
        return outputs, self_attn_probs, self_attention_mask   

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.d_model)

        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs, self_attention_mask):

        outputs, self_attn_prob = self.self_attention(inputs, inputs, inputs, self_attention_mask)
        outputs = self.attention_norm(inputs + outputs)

        inputs = outputs
        outputs = self.feed_forward(inputs)
        outputs = self.feed_forward_norm(inputs + outputs)
        
        return outputs, self_attn_prob

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.sqrt_dim = math.sqrt(config.d_model)
        self.pos_encoding = PositionalEncoding(config.max_dec_len, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_dec_layers)]
        )

        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self,
                input_ids,
                attention_mask=None,
                enc_outputs=None,
                enc_attention_mask=None):

        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()

        word_embeds = self.word_embedding(input_ids) * self.sqrt_dim
        position_embeds = self.pos_encoding[:, :seq_len].to(input_ids.device)

        dec_outputs = word_embeds + position_embeds

        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=True) 

        self_attn_probs, cross_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, cross_attn_prob = layer(inputs=dec_outputs, 
                                                                 self_attention_mask=self_attention_mask, 
                                                                 enc_outputs=enc_outputs, 
                                                                 cross_attention_mask=enc_attention_mask,
                                                                 )
            self_attn_probs.append(self_attn_prob)
            cross_attn_probs.append(cross_attn_prob)        
        
        dec_outputs = self.fc(dec_outputs)

        return dec_outputs, self_attn_probs, cross_attn_probs

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(config)
        self.self_attention_norm = nn.LayerNorm(config.d_model)

        self.cross_attention = MultiHeadAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.d_model)
        
        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, 
        inputs, 
        self_attention_mask,
        enc_outputs,  
        cross_attention_mask):
        
        outputs, self_attn_prob = self.self_attention(inputs, inputs, inputs, self_attention_mask)
        outputs = self.self_attention_norm(inputs + outputs)

        inputs = outputs
        outputs, cross_attn_prob = self.cross_attention(inputs, enc_outputs, enc_outputs, cross_attention_mask)
        outputs = self.cross_attention_norm(inputs + outputs)
        
        inputs = outputs
        outputs = self.feed_forward(inputs)
        outputs = self.feed_forward_norm(inputs + outputs)

        return outputs, self_attn_prob, cross_attn_prob

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        if config.use_decoder == True:
            self.decoder = TransformerDecoder(config)
        
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
                dec_input_ids=None, 
                enc_attention_mask=None,
                dec_attention_mask=None,
                ):
        
        enc_outputs, enc_self_attn_probs, enc_attention_mask = self.encoder(input_ids=enc_input_ids, 
                                                                            attention_mask=enc_attention_mask)
        
        if self.config.use_decoder == False:
            return enc_outputs, enc_self_attn_probs
        
        dec_outputs, dec_self_attn_probs, dec_cross_attn_probs = self.decoder(input_ids=dec_input_ids,
                                                                              attention_mask=dec_attention_mask,
                                                                              enc_outputs=enc_outputs, 
                                                                              enc_attention_mask=enc_attention_mask)
        
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_cross_attn_probs
