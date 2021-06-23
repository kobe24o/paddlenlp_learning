目录：
[1. 导入一些包](导入一些包)
[2. 加载数据](加载数据)
[3. 数据预处理](
  3.1 获取tokenizer，得到 input_ids, token_type_ids
  3.2 转换函数、batch化函数、sampler、data_loader
[4. 编写模型](
[5. 学习率、参数衰减、优化器、loss、评估标准](
[6. 评估函数](
[7. 训练+评估](
[8. 保存模型到文件](
[9. 预测](
[10. 多GPU并行设置](

项目介绍 项目链接：[https://aistudio.baidu.com/aistudio/projectdetail/2029701](https://aistudio.baidu.com/aistudio/projectdetail/2029701)
单机多卡训练参考：[https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/06_device_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/06_device_cn.html)
支持 `star PaddleNLP` github [https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

## 1. 导入一些包

```python
import time
import os
import numpy as np
import paddle
import paddlenlp
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddle.distributed as dist  # 并行
```

## 2. 加载数据

```python
batch_size = 64
epochs = 5

# 加载数据集
train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
# 展示数据
for i, example in enumerate(train_ds):
    if i < 5:
        print(example)
    # {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}
    # {'query': '我手机丢了，我想换个手机', 'title': '我想买个新手机，求推荐', 'label': 1}
    # {'query': '大家觉得她好看吗', 'title': '大家觉得跑男好看吗？', 'label': 0}
    # {'query': '求秋色之空漫画全集', 'title': '求秋色之空全集漫画', 'label': 1}
    # {'query': '晚上睡觉带着耳机听音乐有什么害处吗？', 'title': '孕妇可以戴耳机听音乐吗?', 'label': 0}
```

## 3. 数据预处理
### 3.1 获取tokenizer，得到 input_ids, token_type_ids
```python
# 使用预训练模型的tokenizer
tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained("ernie-gram-zh")
# https://gitee.com/paddlepaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst


def convert_data(data, tokenizer, max_seq_len=512, is_test=False):
    text1, text2 = data["query"], data["title"]
    encoded_inputs = tokenizer(text=text1, text_pair=text2, max_seq_len=max_seq_len)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([data["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    return input_ids, token_type_ids


input_ids, token_type_ids, label = convert_data(train_ds[0], tokenizer)
print(input_ids)
# [1, 692, 811, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2, 
#  329, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2]
print(token_type_ids)
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(label)
# [1]
```
### 3.2 转换函数、batch化函数、sampler、data_loader
- 包装转换函数，方便简化后续代码

```python
from functools import partial
trans_func = partial(convert_data, tokenizer=tokenizer, max_seq_len=512)
```
- 生成 data_loader
```python
from paddlenlp.data import Stack, Pad, Tuple

# batch 化函数
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    Stack(dtype="int64") # 分别对应于 input_ids, token_type_ids, label
): [d for d in fn(samples)]
# 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
# 长度是指的： 一个batch中的最大长度，主要考虑性能开销
# paddlenlp.data.Tuple	将多个batchify函数包装在一起

batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)
# 注意训练可以用 用分布式的 sampler，充分利用资源

train_data_loader = paddle.io.DataLoader(
    dataset=train_ds.map(trans_func), # 数据转换
    batch_sampler=batch_sampler, # 取样
    collate_fn=batchify_fn, # batch化函数
    return_list=True
)

batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
    dataset=dev_ds.map(trans_func),
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True
)
```

## 4. 编写模型
预训练模型，接 FC

```python
import paddle.nn as nn
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained("ernie-gram-zh")


# %%

class TeachingPlanModel(nn.Layer):
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.clf = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.clf(cls_embedding)
        probs = F.softmax(logits)
        return probs


model = TeachingPlanModel(pretrained_model)
```

## 5. 学习率、参数衰减、优化器、loss、评估标准

```python
from paddlenlp.transformers import LinearDecayWithWarmup

num_training_steps = len(train_data_loader) * epochs

# 学习率调度器
lr_scheduler = LinearDecayWithWarmup(5e-5, num_training_steps, 0.0)
# 衰减的参数
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params
)

# 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估标准
metric = paddle.metric.Accuracy()
```
## 6. 评估函数

```python
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        # 前向传播
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        # 损失
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        # 准确率
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
    print("评估 {} loss: {:.5}, acc: {:.5}".format(phase, np.mean(losses), acc))
    model.train()
    metric.reset()
```

## 7. 训练+评估

```python
global global_step
global_step = 0
t_start = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 前向传播
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        # 损失
        loss = criterion(probs, labels)
        # 准确率
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
        
        global_step += 1
		
		# 打印训练信息
        if global_step % 10 == 0:
            print("训练步数 %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                  % (global_step, epoch, step, loss, acc,
                     10 / (time.time() - t_start)))
            t_start = time.time()
		# 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        # 清除梯度
        optimizer.clear_grad()
		
		# 训练100步，评估一次
        if global_step % 100 == 0:
            evaluate(model, criterion, metric, dev_data_loader, "dev")
```

训练过程：

```python
训练步数 5010, epoch: 3, batch: 1278, loss: 0.39062, acc: 0.90781, speed: 0.33 step/s
训练步数 5020, epoch: 3, batch: 1288, loss: 0.41552, acc: 0.90312, speed: 1.87 step/s
训练步数 5030, epoch: 3, batch: 1298, loss: 0.34011, acc: 0.90521, speed: 1.57 step/s
训练步数 5040, epoch: 3, batch: 1308, loss: 0.37718, acc: 0.90703, speed: 1.55 step/s
训练步数 5050, epoch: 3, batch: 1318, loss: 0.35848, acc: 0.91125, speed: 1.80 step/s
训练步数 5060, epoch: 3, batch: 1328, loss: 0.37751, acc: 0.91042, speed: 1.67 step/s
训练步数 5070, epoch: 3, batch: 1338, loss: 0.42495, acc: 0.91161, speed: 1.72 step/s
训练步数 5080, epoch: 3, batch: 1348, loss: 0.38556, acc: 0.91035, speed: 1.67 step/s
训练步数 5090, epoch: 3, batch: 1358, loss: 0.40671, acc: 0.91024, speed: 1.85 step/s
训练步数 5100, epoch: 3, batch: 1368, loss: 0.36824, acc: 0.91000, speed: 1.74 step/s
评估 dev loss: 0.44395, acc: 0.86321
训练步数 5110, epoch: 3, batch: 1378, loss: 0.41520, acc: 0.92188, speed: 0.32 step/s
训练步数 5120, epoch: 3, batch: 1388, loss: 0.42261, acc: 0.91250, speed: 1.65 step/s
训练步数 5130, epoch: 3, batch: 1398, loss: 0.37139, acc: 0.91615, speed: 1.68 step/s
训练步数 5140, epoch: 3, batch: 1408, loss: 0.38124, acc: 0.90781, speed: 1.68 step/s
训练步数 5150, epoch: 3, batch: 1418, loss: 0.41482, acc: 0.90781, speed: 1.76 step/s
训练步数 5160, epoch: 3, batch: 1428, loss: 0.38554, acc: 0.91120, speed: 1.75 step/s
训练步数 5170, epoch: 3, batch: 1438, loss: 0.38424, acc: 0.91027, speed: 1.77 step/s
训练步数 5180, epoch: 3, batch: 1448, loss: 0.39620, acc: 0.90938, speed: 1.72 step/s
训练步数 5190, epoch: 3, batch: 1458, loss: 0.41320, acc: 0.90747, speed: 1.77 step/s
训练步数 5200, epoch: 3, batch: 1468, loss: 0.39017, acc: 0.90859, speed: 1.64 step/s
评估 dev loss: 0.4526, acc: 0.8556
```

## 8. 保存模型到文件

```python
pathname = "checkpoint"
isExists = os.path.exists(pathname)
if not isExists:
    os.mkdir(pathname)
    
save_dir = os.path.join(pathname, "model_%d" % global_step)
save_param_path = os.path.join(save_dir, "model_state_pdparams")

paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)
```

## 9. 预测

```python
def predict(model, data_loader):
    batch_probs = []
    model.eval() # 评估模式
    with paddle.no_grad(): # 不需要梯度更新
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)
        return batch_probs

# 数据转换函数
trans_func_test = partial(convert_data, tokenizer=tokenizer, max_seq_len=512, is_test=True)

# batch化函数
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
): [data for data in fn(samples)]

# 加载测试集
test_ds = load_dataset("lcqmc", splits=["test"])

# 定义 sampler
batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=batch_size, shuffle=False)

# 定义data_loader
predict_data_loader = paddle.io.DataLoader(
    dataset=test_ds.map(trans_func_test),
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True
)

# 定义模型
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained("ernie-gram-zh")
model = TeachingPlanModel(pretrained_model)

# 加载训练好的参数
state_dict = paddle.load(save_param_path)
# 设置参数
model.set_dict(state_dict)

# 预测
y_probs = predict(model, predict_data_loader)
y_preds = np.argmax(y_probs, axis=1)

# 预测结果写入文件
with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
    f.write("index\tprediction\n")
    for idx, y_pred in enumerate(y_preds):
        f.write("{}\t{}\n".format(idx, y_pred))
        # text_pair = test_ds.data[idx]
        # text_pair["label"] = y_pred
        # print(text_pair)
```

## 10. 多GPU并行设置

```python
import paddle.distributed as dist  # 并行

if __name__ == "__main__":
    dist.init_parallel_env()  # 初始化并行环境
    # 启动命令 python -m paddle.distributed.launch --gpus '0,1' xxx.py &
    # your code 。。。
```
可以看见 2个 GPU 都使用起来了

```python
Sat Jun 19 18:18:34 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA Tesla T4     Off  | 00000000:00:09.0 Off |                    0 |
| N/A   67C    P0    69W /  70W |   9706MiB / 15109MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA Tesla T4     Off  | 00000000:00:0A.0 Off |                    0 |
| N/A   68C    P0    68W /  70W |  11004MiB / 15109MiB |     99%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     34450      C   ...nda3/envs/pp21/bin/python     9703MiB |
|    1   N/A  N/A     34453      C   ...nda3/envs/pp21/bin/python    11001MiB |
+-----------------------------------------------------------------------------+
```
