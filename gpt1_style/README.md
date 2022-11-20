# GPT1

## 특징
- Deocder 모델
- Transformer의 Decoder에서 Cross attention 등 Encoder와 상관이 있는 부분이 제외됨
- input은 Bert와 같이 input_ids, token_type_ids, attention_mask를 사용. Bert의 경우에는 Token_type_ids와 word_embedding을 별도의 embedding으로 나눠 사용하는데, GPT1의 경우에는 word_embedding에 그대로 Token_type_ids 값을 넣어서 사용하는 점이 특이함
- 초기의 Transformer 구현 코드에서 자주 보인 1D Conv가 Linear 대신에 사용 됨
- 특히 querty, key, value를 구하는 방식이 다른 요즘의 Transformer 모델들과 달리 input의 last dim을 3*d_model로 늘린 후 split하는 구조를 취하고 있음
