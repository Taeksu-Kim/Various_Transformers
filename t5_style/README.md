# T5 Style

## 특징
- Encoder-Decoder 모델
- Position Embedding : 상대위치 position embedding을 position bias로 사용. 
- Token_type_embedding을 사용하지 않음. word embedding만 구하고 attention 부분에서 position bais를 attention score matrix에 더 해줌(compute_bias 참고)
- Encoder의 Self attention시에는 양방향 position bias 사용, Decoder의 Self attention에서는 단방향 position bias, Cross attention시에는 position bais를 사용하지 않음
- Transformer, 다른 모델들과 달리 attention 계산에 scale 값을 곱하거나 나누지 않음
- Transformer, Bert와 달리 Attention에서 query에만 layer_norm을 진행하고 key, value는 layer_norm이 되지 않은 값을 인풋으로 받음
- Feedforward의 Activation function을 gelu로 사용한 Bert와 달리 Transformer와 같이 relu 사용 
- 모든 Linear에 bias=False이 적용되어 있음
- Add&Norm 방식이 GPT2와 유사하게 residual에는 layer_norm이 적용이 되지 않도록 돼있음. GPT2와 달리 T5는 Cross attention이 있고, Self attention에는 query, key, value 모두 layer_norm을 통과한 값을 사용하지만 Cross attention에서는 key, value로 layer_norm을 통과하지 않은 값을 사용하는 점에서 구조적 차이가 있음.

- 적절한 prefix를 추가하여 사용(더 깊게 알아봐야할 듯)
