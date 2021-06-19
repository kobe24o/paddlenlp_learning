# 数据集 https://aistudio.baidu.com/aistudio/competition/detail/50
import os
import time

import numpy as np
from functools import partial
import paddle
import paddlenlp
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddle.distributed as dist
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
from my_utils import create_dataloader, convert_example, evaluate
from paddlenlp.data import Tuple, Pad, Stack
import paddle.distributed as dist  # 并行

batch_size = 16
max_seq_len = 128
epochs = 5


def read(data_path, flag=0):
    with open(data_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头列名
        for line in f:
            if flag == 0:
                label, text = line.strip().split("\t")
            elif flag == 1:
                id, label, text = line.strip().split("\t")
            else:
                id, text = line.strip().split("\t")
            if flag == 0 or flag == 1:
                yield {"label": label, "text": text}
            else:
                yield {"id": id, "text": text}


# 训练
def train(model, train_data_loader, criterion, metric, optimizer):
    global_step = 0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, step, loss, acc, 10 / (time.time() - start_time)))
                start_time = time.time()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % 100 == 0:
                save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                evaluate(model, criterion, metric, dev_data_loader)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    dist.init_parallel_env()  # 初始化并行环境
    # 启动命令 python -m paddle.distributed.launch --gpus '0,1' sentimentClassification.py &
    train_data = "./ChnSentiCorp/train.tsv"
    dev_data = "./ChnSentiCorp/dev.tsv"
    test_data = "./ChnSentiCorp/test.tsv"
    train_data = load_dataset(read, data_path=train_data, lazy=False)
    dev_data = load_dataset(read, data_path=dev_data, flag=1, lazy=False)
    test_data = load_dataset(read, data_path=test_data, flag=2, lazy=False)
    # 创建数据迭代器

    # for d in dev_data:
    #     print(d["label"], d["text"]) # 展示数据

    # 模型, tokenizer
    model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch",
                                                          num_classes=2)
    # https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/skep
    tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")

    # 数据转换函数，转换为模型可以读入的数据
    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=max_seq_len)

    # 数据组成批量，padding，label堆叠
    batchify_fn = lambda samples, fn=Tuple(
        Pad(pad_val=tokenizer.pad_token_id, axis=0),
        Pad(pad_val=tokenizer.pad_token_type_id, axis=0),
        Stack()
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(train_data, mode="train", batch_size=batch_size, batchify_fn=batchify_fn,
                                          trans_func=trans_func)
    dev_data_loader = create_dataloader(dev_data, mode="dev", batch_size=batch_size, batchify_fn=batchify_fn,
                                        trans_func=trans_func)

    ckpt_dir = "skep_ckpt"
    num_train_steps = len(train_data_loader) * epochs
    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    train(model, train_data_loader, criterion, metric, optimizer)

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=max_seq_len, is_test=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(pad_val=tokenizer.pad_token_id, axis=0),
        Pad(pad_val=tokenizer.pad_token_type_id, axis=0),
        Stack()
    ): [data for data in fn(samples)]
    test_data_loader = create_dataloader(test_data, mode="test", batch_size=batch_size, batchify_fn=batchify_fn,
                                         trans_func=trans_func)

    params_path = 'skep_ckpt/model_500/model_state.pdparams'
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("从 {} 加载模型参数".format(params_path))

    label_map = {0: '0', 1: '1'}
    ans = []

    model.eval()
    for batch in test_data_loader:
        input_ids, token_type_ids, qids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=-1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        qids = qids.numpy().tolist()
        ans.extend(zip(qids, labels))

    res_dir = "./results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # 写入预测结果
    with open(os.path.join(res_dir, "ChnSentiCorp.tsv"), 'w', encoding="utf-8") as f:
        f.write("index\tprediction\n")
        for qid, label in ans:
            f.write(str(qid[0]) + "\t" + label + "\n")
