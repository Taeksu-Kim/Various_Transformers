# Original Transformer

## 특징
- Position embedding : Positional Encoding 사용
- Encoder-Decoder Model
- Encoder는 Bidirectional, Decoder는 Autoregressive 방식을 사용
- Decoder는 Self Attention과 Cross Attention(Encoder Decoder Attention) 수행
- Decoder는 Autoregressive하기 때문에 Self Attention 수행시 해당 토큰 뒤를 Mask하는 Subsequent_mask도 적용
- 스케일링 값으로는 (hidden_dim / num_attention_head) ** 0.5로 d_head의 루트값을 사용
- 스케일링은 word embedding * scale, torch.matmul(query, key.transpose(-2, -1)) / self.scale로 스케일링 수행
