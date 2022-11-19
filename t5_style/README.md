# T5 Style

## 특징
- Encoder-Decoder 모델
- Position Embedding : 상대위치 position embedding을 position bias로 사용. 
- Token_type_embedding을 사용하지 않음. word embedding만 구하고 attention 부분에서 position bais를 attention score matrix에 더 해줌(compute_bias 참고)
- Encoder의 Self attention시에는 양방향 position bias 사용, Decoder의 Self attention에서는 단방향 position bias, Cross attention시에는 position bais를 사용하지 않음
- Transformer, Bert와 달리 별도의 embedding값이나 attention 계산에 scale 값을 곱하거나 나누지 않음
- Transformer, Bert와 달리 Attention에서 query에만 layer_norm을 진행하고 key, value는 layer_norm이 되지 않은 값을 인풋으로 받음
- Feedforward의 Activation function을 gelu로 사용한 Bert와 달리 Transformer와 같이 relu 사용 
- 모든 Linear에 bias=False이 적용되어 있음

- songys  Chatbot data로 pretrain 없이 학습시켰을 때 거의 학습이 되지 않음. 구현상의 미스인가 해서 허깅페이스 T5로 돌려보더라도 마찬가지 결과. 바닐라 트랜스포머나 Bert의 경우에는 pretrain 없이도 어느정도 학습이 됐었음. position embedding의 차이 혹은 데이터의 길이가 너무 짧아서 그런것으로 보임.
