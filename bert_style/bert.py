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

class PoswiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForward, self).__init__()      

        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.feed_forward_dim),
                                          nn.GELU(),
                                          nn.Linear(config.feed_forward_dim, config.d_model),
                                          nn.Dropout(config.drop_out_raito))
    def forward(self, inputs):
        return self.feed_forward(inputs)
    
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)

        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.drop_out_raito)

        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))

    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None,
        ):
        
        batch_size, seq_len = input_ids.size()

        inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_ids = self.position_ids[:, :seq_len].to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
class BertEncoder(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.num_enc_layers)]
        )

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                ):

        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).int()
        if token_type_ids is None:
            token_type_ids = torch.zeros([batch_size, seq_len], dtype=torch.long, device=device)

        outputs = self.embedding(input_ids,
                                 token_type_ids=token_type_ids)
        
        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=False)

        self_attn_probs = []
        for i, layer in enumerate(self.layers):
            outputs, self_attn_prob = layer(inputs=outputs,
                                            self_attention_mask=self_attention_mask,
                                            )
            self_attn_probs.append(self_attn_prob)

        return outputs, self_attn_probs, self_attention_mask    
    
class BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = BertAttention(config)
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self, 
        inputs, 
        self_attention_mask,
        ):

        outputs, self_attn_prob = self.self_attention(inputs, inputs, inputs, self_attention_mask)
        outputs = self.attention_norm(inputs + outputs)

        inputs = outputs
        outputs = self.feed_forward(inputs)
        outputs = self.feed_forward_norm(inputs + outputs)
        
        return outputs, self_attn_prob

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_att_heads = config.num_att_heads
        assert self.d_model % self.num_att_heads == 0, "d_model({}) % num_att_heads({}) = {}. It should be 0.".format(self.d_model, self.num_att_heads, self.d_model % self.num_att_heads)
        self.d_head = int(self.d_model / self.num_att_heads)
        self.scale = self.d_head ** 0.5
        
        self.query_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.key_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)
        self.value_proj = nn.Linear(self.d_model, self.num_att_heads * self.d_head)

        self.attn_dropout = nn.Dropout(config.drop_out_raito)

        self.fc = nn.Linear(self.d_head * self.num_att_heads, self.d_model)
        self.context_dropout = nn.Dropout(config.drop_out_raito)

    def forward(self,
                query,
                key,
                value,
                attention_mask,
                ):
      
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, query_len, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, key_len, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_att_heads, self.d_head).transpose(1,2) # [bs, num_heads, value_len, d_head]

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [bs, num_heads, query_len, key_len]        
        scores = scores + attention_mask
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.attn_dropout(attn_prob)

        context = torch.matmul(attn_prob, value) # [bs, num_heads, query_len, d_head]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_att_heads * self.d_head)
        
        context = self.fc(context)
        context = self.context_dropout(context)

        return context, attn_prob

class BertModel(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.embedding = BertEmbeddings(config)
      self.encoder = BertEncoder(config, self.embedding)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                ):

      outputs, self_attn_probs, _ = self.encoder(input_ids=input_ids,
                                                 token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask,
                                                 )
      
      return outputs, self_attn_probs

class BertDecoder(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [BertDecoderLayer(config) for _ in range(config.num_enc_layers)]
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
        token_type_ids = torch.zeros([batch_size, seq_len], dtype=torch.long, device=input_ids.device)

        outputs = self.embedding(input_ids,
                                 token_type_ids=token_type_ids)

        self_attention_mask = get_extended_attention_mask(attention_mask, autoregressive=True)

        self_attn_probs, cross_attn_probs = [], []
        for i, layer in enumerate(self.layers):
            outputs, self_attn_prob, cross_attn_prob = layer(inputs=outputs,
                                                             self_attention_mask=self_attention_mask,
                                                             enc_outputs=enc_outputs,
                                                             cross_attention_mask=enc_attention_mask,
                                                             )
            self_attn_probs.append(self_attn_prob)
            cross_attn_probs.append(cross_attn_prob)      

        outputs = self.fc(outputs)        

        return outputs, self_attn_probs, cross_attn_probs

class BertDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = BertAttention(config)
        self.self_attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.cross_attention = BertAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.feed_forward = PoswiseFeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
    def forward(self,
                inputs,
                self_attention_mask,
                enc_outputs,
                cross_attention_mask,
                ):

        outputs, self_attn_prob = self.self_attention(inputs, inputs, inputs, self_attention_mask)
        outputs = self.self_attention_norm(inputs + outputs)

        inputs = outputs
        outputs, cross_attn_prob = self.cross_attention(inputs, enc_outputs, enc_outputs, cross_attention_mask)
        outputs = self.cross_attention_norm(inputs + outputs)

        inputs = outputs
        outputs = self.feed_forward(inputs)
        outputs = self.feed_forward_norm(inputs + outputs)
        
        return outputs, self_attn_prob, cross_attn_prob

class Bert_Encoder_Decoder_Model(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.config=config

      if config.share_embedding is True:
          self.shared_embedding = BertEmbeddings(config)
          self.encoder = BertEncoder(config, self.shared_embedding)
          self.decoder = BertDecoder(config, self.shared_embedding)
      
      else:
          self.encoder_embedding = BertEmbeddings(config)
          self.decoder_embedding = BertEmbeddings(config)
  
          self.encoder = BertEncoder(config, self.encoder_embedding)
          self.decoder = BertDecoder(config, self.decoder_embedding)

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
                enc_token_type_ids=None,
                enc_attention_mask=None,
                dec_input_ids=None,
                dec_attention_mask=None,
                ):

        enc_outputs, enc_self_attn_probs, enc_attention_mask = self.encoder(enc_input_ids,
                                                                  enc_token_type_ids,
                                                                  enc_attention_mask,
                                                                  )
        
        dec_outputs, dec_self_attn_probs, dec_cross_attn_probs = self.decoder(input_ids=dec_input_ids,
                                                                              attention_mask=dec_attention_mask,
                                                                              enc_outputs=enc_outputs,
                                                                              enc_attention_mask=enc_attention_mask,
                                                                              )

        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_cross_attn_probs
