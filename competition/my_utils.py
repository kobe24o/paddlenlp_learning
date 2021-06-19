import numpy as np
import paddle


def create_dataloader(dataset, trans_func=None, mode="train", batch_size=1, batchify_fn=None):
    # 生成 batch 数据喂入模型
    if trans_func:
        dataset = dataset.map(trans_func)
    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset,batch_sampler=sampler,collate_fn=batchify_fn)
    return dataloader

def convert_example(example, tokenizer, max_seq_len=512, is_test=False):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_len)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        qid = np.array([example["id"]], dtype="int64")
        return input_ids, token_type_ids, qid

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        acc = metric.accumulate()
    print("评估损失：{:.5f}, 准确率：{:.5f}".format(np.mean(losses), acc))
    model.train()
    metric.reset()