# Original Transformer

## 특징
- Encoder-Decoder 모델
- Position embedding : Positional Encoding 사용
- Encoder는 Bidirectional, Decoder는 Autoregressive 방식을 사용
- Decoder는 Self Attention과 Cross Attention(Encoder Decoder Attention) 수행
- Decoder는 Autoregressive하기 때문에 Self Attention 수행시 해당 토큰 뒤를 Mask하는 Subsequent_mask도 적용
- 스케일링 값으로는 (hidden_dim / num_attention_head) ** 0.5로 d_head의 루트값을 사용
- 스케일링은 word embedding * scale, torch.matmul(query, key.transpose(-2, -1)) / self.scale로 스케일링 수행
- attention에 scale값을 곱하는 것은 T5를 제외한 다른 모델들에서도 공통적이지만 word embedding에 scale을 곱하는 것은 바닐라 트랜스포머만 해당함
