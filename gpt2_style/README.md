# GPT2

## 특징
- Decoder 모델
- GPT1과 같이 querty, key, value 및 feed_forward부분에서 linear가 아니라 1D conv 레이어 사용
- Transformer의 Add&Norm에 변화를 줌. 기존: layer_norm(x+sub_layer(x)) -> GPT2: x+sub_layer(layer_nrom(x))
- GPT1에 비해 전체적인 모델의 크기가 커지가 되지만 이는 파라미터 조정에 따른 것으로 모델의 구조적인 변경점은 layer_norm 이외에는 크게 보이지 않음
