#%%

import time
import os
import numpy as np
import paddle
import paddlenlp
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddle.distributed as dist #并行

if __name__ == "__main__":
    dist.init_parallel_env()  # 初始化并行环境
    # 启动命令 python -m paddle.distributed.launch --gpus '0,1' xxx.py &
    batch_size=64
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

    #%%
    # 数据预处理
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
    print(token_type_ids)
    print(label)

    from functools import partial
    trans_func = partial(convert_data, tokenizer=tokenizer, max_seq_len=512)

    from paddlenlp.data import Stack, Pad, Tuple
    batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64")
    ) : [d for d in fn(samples)]
    # 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
    # 长度是指的： 一个batch中的最大长度，主要考虑性能开销

    batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)

    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True
    )

    batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=batch_size, shuffle=False)
    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True
    )

    #%%
    import paddle.nn as nn
    pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained("ernie-gram-zh")
    #%%

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
    # 并行训练设置
    # https://aistudio.baidu.com/aistudio/projectdetail/1222066
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/06_device_cn.html

    #%%
    # 训练评估
    from paddlenlp.transformers import LinearDecayWithWarmup
    num_training_steps = len(train_data_loader)*epochs
    lr_scheduler = LinearDecayWithWarmup(5e-5, num_training_steps, 0.0)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x : x in decay_params
    )

    criterion = paddle.nn.loss.CrossEntropyLoss()

    metric = paddle.metric.Accuracy()
    #%%
    @paddle.no_grad()
    def evaluate(model, criterion, metric, data_loader, phase="dev"):
        model.eval()
        metric.reset()
        losses = []
        for batch in data_loader:
            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, labels)
            losses.append(loss.numpy())
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()
        print("评估 {} loss: {:.5}, acc: {:.5}".format(phase, np.mean(losses), acc))
        model.train()
        metric.reset()

    global global_step
    global_step = 0
    t_start = time.time()
    for epoch in range(1, epochs+1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()
            global_step += 1
            # if global_step > 10 :
            #     break # 调试代码

            if global_step%10 == 0:
                print("训练步数 %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                        10 / (time.time() - t_start)))
                t_start = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step%100 == 0:
                evaluate(model, criterion, metric, dev_data_loader, "dev")
    #%%
    pathname = "checkpoint"
    isExists=os.path.exists(pathname)
    if not isExists:
        os.mkdir(pathname)
    save_dir = os.path.join(pathname, "model_%d" % global_step)
    save_param_path = os.path.join(save_dir, "model_state_pdparams")
    paddle.save(model.state_dict(), save_param_path)
    tokenizer.save_pretrained(save_dir)
    #%%
    #预测
    def predict(model, data_loader):
        batch_probs = []
        model.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
                batch_probs.append(batch_prob)
            batch_probs = np.concatenate(batch_probs, axis=0)
            return batch_probs

    trans_func_test = partial(convert_data, tokenizer=tokenizer, max_seq_len=512, is_test=True)

    batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    ) : [data for data in fn(samples)]

    test_ds = load_dataset("lcqmc", splits=["test"])

    batch_sampler = paddle.io.BatchSampler(test_ds,batch_size=batch_size, shuffle=False)
    predict_data_loader = paddle.io.DataLoader(
        dataset=test_ds.map(trans_func_test),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True
    )

    #%%
    pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained("ernie-gram-zh")
    model = TeachingPlanModel(pretrained_model)
    state_dict = paddle.load(save_param_path)
    model.set_dict(state_dict)

    for idx, batch in enumerate(predict_data_loader):
        if idx < 1:
            print(batch)
    y_probs = predict(model, predict_data_loader)
    y_preds = np.argmax(y_probs, axis=1)
    # y_preds = [0 for _ in range(len(predict_data_loader))]
    with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
        f.write("index\tprediction\n")
        for idx, y_pred in enumerate(y_preds):
            f.write("{}\t{}\n".format(idx, y_pred))
            # text_pair = test_ds.data[idx]
            # text_pair["label"] = y_pred
            # print(text_pair)
