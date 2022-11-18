# T5 Style

## 특징
- Encoder-Decoder 모델
- Position Embedding : 상대위치 position embedding을 position bias로 사용. 
- Token_type_embedding을 사용하지 않음. word embedding만 구하고 attention 부분에서 position bais를 attention score matrix에 더 해줌(_compute_bias 참고)
- Transformer, Bert와 달리 별도의 scale 값을 곱하거나 나누지 않음
- Transformer, Bert와 달리 Attention에서 query에만 layer_norm을 진행하고 key, value는 layer_norm이 되지 않은 값을 인풋으로 받음
- Feedforward의 Activation function을 gelu로 사용한 Bert와 달리 Transformer와 같이 relu 사용 
