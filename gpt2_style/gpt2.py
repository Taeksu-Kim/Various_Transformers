import math
import torch
import torch.nn as nn

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

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForward, self).__init__()      

        self.feed_forward = nn.Sequential(Conv1D(config.d_model * 4, config.d_model),
                                          nn.GELU(),
                                          Conv1D(config.d_model, config.d_model * 4),
                                          nn.Dropout(config.drop_out_raito))
    def forward(self, inputs):
        return self.feed_forward(inputs)

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        self.d_head = int(self.d_model / self.num_att_heads)
        self.scale = self.d_head ** 0.5

        self.conv_layer = Conv1D(self.d_model * 3, self.d_model)
        self.attn_dropout = nn.Dropout(config.drop_out_raito)
        
        self.fc = Conv1D(self.d_model, self.d_model)
        self.context_dropout = nn.Dropout(config.drop_out_raito)

    def forward(self,
                query,
                key=None,
                value=None,
                attention_mask=None,
                ):

        if key is None and value is None:
            query = self.conv_layer(query)
            query, key, value = query.split(self.d_model, dim=2)

        batch_size = query.size(0)

        query =  query.view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = key.view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = value.view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [bs, num_heads, query_len, key_len]        
        scores = scores + attention_mask
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.attn_dropout(attn_prob)

        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        context = self.fc(context)
        context = self.context_dropout(context)

        return context, attn_prob

class GPT2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.dropout = nn.Dropout(config.drop_out_raito)
        
        self.layers = nn.ModuleList(
            [GPT2DecoderLayer(config) for i in range(config.num_dec_layers)]
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.fc = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        ):

        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()

        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, seq_len], dtype=torch.long, device=input_ids.device)

        position_ids = self.position_ids[:, :seq_len].to(input_ids.device)

        word_embeds = self.word_embedding(input_ids)
        token_type_embeds = self.word_embedding(token_type_ids)
        position_embeds = self.position_embedding(position_ids)

        outputs = word_embeds + token_type_embeds + position_embeds
        outputs = self.dropout(outputs)

        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=True)

        self_attn_probs = []
        for i, layer in enumerate(self.layers):
            outputs, self_attn_prob = layer(inputs=outputs,
                                            self_attention_mask=self_attention_mask, 
                                            )
            self_attn_probs.append(self_attn_prob)     

        outputs = self.layer_norm(outputs)

        outputs = self.fc(outputs)
 
        return outputs, self_attn_probs

class GPT2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = GPT2Attention(config)
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)     

    def forward(self,
                inputs,
                self_attention_mask,
                ):
        outputs, self_attn_prob = self.self_attention(query=self.attention_norm(inputs),
                                                      key=None,
                                                      value=None, 
                                                      attention_mask=self_attention_mask,
                                                      )
        outputs = inputs + outputs
        
        inputs = outputs
        outputs = self.feed_forward(self.feed_forward_norm(inputs))
        outputs = inputs + outputs
        
        return outputs, self_attn_prob
