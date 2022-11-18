# Bert Style

## 특징
- Position Embedding : 절대위치 position embedding
- Embedding
  - 초기 트랜스포머 : Word_embedding + Positional encoding을 사용
  - Bert : Word_embedding + Token_type_embedding(인풋A(0), 인풋B(1) 구분) + Position_embedding 사용
- Encoder 모델(Decoder도 Transformer 형식으로 붙일 수는 있으나 Bert 이후로는 PLM으로 자주 사용하는 것을 고려하면 애매한 부분이 있음)
- Activation Function이 Relu -> Gelu로 변경됨
