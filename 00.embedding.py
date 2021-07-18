from paddlenlp.embeddings import TokenEmbedding
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")
# 可以选择不同的 embeddings
# 见 https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/embeddings/constant.py
print(token_embedding)
# Unknown index: 635963
# Unknown token: [UNK]
# Padding index: 635964
# Padding token: [PAD]
# Shape :[635965, 300]
# Object   type: TokenEmbedding(635965, 300, padding_idx=635964, sparse=False)
# Unknown index: 635963
# Unknown token: [UNK]
# Padding index: 635964
# Padding token: [PAD]
# Parameter containing:
# Tensor(shape=[635965, 300], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
#        [[-0.24200200,  0.13931701,  0.07378800, ...,  0.14103900,  0.05592300, -0.08004800],
#         [-0.08671700,  0.07770800,  0.09515300, ...,  0.11196400,  0.03082200, -0.12893000],
#         [-0.11436500,  0.12201900,  0.02833000, ...,  0.11068700,  0.03607300, -0.13763499],
#         ...,
#         [ 0.02628800, -0.00008300, -0.00393500, ...,  0.00654000,  0.00024600, -0.00662600],
#         [-0.00537162, -0.01470578,  0.03992381, ...,  0.02658029, -0.02127731, -0.02485077],
#         [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]])

# 获得指定词汇的词向量
test_token_embedding = token_embedding.search("自然语言处理")
print(test_token_embedding)
