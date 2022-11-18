# Bert Style

## 특징
- Encoder 모델(Decoder도 Transformer와 마찬가지로 붙일 수는 있으나 Bert 이후로는 PLM으로 자주 사용하기 때문에 굳이 쓸 이유가 애매)
- Position Embedding : 절대위치 position embedding
- Embedding
  - 초기 트랜스포머 : Word_embedding + Positional encoding을 사용
  - Bert : Word_embedding + Token_type_embedding(인풋A(0), 인풋B(1) 구분) + Position_embedding 사용
- Activation Function이 Relu -> Gelu로 변경됨
