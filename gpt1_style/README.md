# GPT1

## 특징
- Deocder 모델
- Transformer의 Decoder에서 Cross attention 등 Encoder와 상관이 있는 부분이 제외됨
- 초기의 Transformer 구현 코드에서 자주 보인 1D Conv가 Linear 대신에 사용 됨
- 특히 querty, key, value를 구하는 방식이 다른 요즘의 Transformer 모델들과 달리 input의 last dim을 3*d_model로 늘린 후 split하는 구조를 취하고 있음
- Transforemr 구조에서 Encoder 관련 부분이 싹 다 빠진 것, 1D Conv가 사용된 것, query, key, value 외에는 특별히 주목할 점은 없는 것으로 보임 
