# Bert Style

## 특징
- Position Embedding : 절대위치 position embedding
- Embedding
  - 초기 트랜스포머 : Word_embedding + Positional encoding을 사용
  - Bert : Word_embedding + Token_type_embedding(인풋A(0), 인풋B(1) 구분) + Position_embedding 사용
- Encoder 모델(Decoder도 추가는 가능)
- Activation Function Relu -> Gelu로 변경
