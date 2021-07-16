# 快递单信息抽取
from functools import partial
import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from paddle.utils.download import get_path_from_url

URL = "https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/waybill.tar.gz"
get_path_from_url(URL, "./")
epochs = 10
batch_size = 16


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab


with open("./data/test.txt", 'r', encoding='utf-8') as f:
    i = 0
    for line in f:
        print(line)
        i += 1
        if i > 5:
            break


# text_a	label
#
# 黑龙江省双鸭山市尖山区八马路与东平行路交叉口北40米韦业涛18600009172
# A1-BA1-IA1-IA1-IA2-BA2-IA2-IA2-IA3-BA3-IA3-IA4-BA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IA4-IP-BP-IP-IT-BT-IT-IT-IT-IT-IT-IT-IT-IT-IT-I
# A1 表示省，-B 开始， -I 内部， P 人名， T 电话

def convert_example(example, tokenizer, label_vocab):
    tokens, labels = example
    tokenized_input = tokenizer(
        tokens, return_length=True, is_split_into_words=True)
    # Token '[CLS]' and '[SEP]' will get label 'O'
    labels = ['O'] + labels + ['O']  # 大写的字母 O（欧）
    tokenized_input['labels'] = [label_vocab[x] for x in labels]
    return tokenized_input['input_ids'], tokenized_input['token_type_ids'], \
           tokenized_input['seq_len'], tokenized_input['labels']
    # 转成数字list


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))
    model.train()


def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds


def parse_decodes(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                # ['1', '6', '6', '2', '0', '2', '0', '0', '0', '7', '7',
                # '宣', '荣', '嗣',
                # '甘', '肃', '省',
                # '白', '银', '市',
                # '会', '宁', '县',
                # '河', '畔', '镇', '十', '字', '街', '金', '海', '超', '市', '西', '行', '5', '0', '米']
                labels = labels.split('\002')
                # ['T-B', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I', 'T-I',
                # 'P-B', 'P-I', 'P-I',
                # 'A1-B', 'A1-I', 'A1-I',
                # 'A2-B', 'A2-I', 'A2-I',
                # 'A3-B', 'A3-I', 'A3-I',
                # 'A4-B', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I', 'A4-I']
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


# Create dataset, tokenizer and dataloader.
train_ds, dev_ds, test_ds = load_dataset(datafiles=(
    './data/train.txt', './data/dev.txt', './data/test.txt'))

label_vocab = load_dict('./data/tag.dic')
# {'P-B': 0, 'P-I': 1, 'T-B': 2, 'T-I': 3, 'A1-B': 4, 'A1-I': 5,
# 'A2-B': 6, 'A2-I': 7, 'A3-B': 8, 'A3-I': 9, 'A4-B': 10, 'A4-I': 11, 'O': 12}
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)
test_ds.map(trans_func)
print(train_ds[0])

ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype='int64'),  # seq_len
    Pad(axis=0, pad_val=ignore_label, dtype='int64')  # labels
): fn(samples)

train_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_size=batch_size,
    return_list=True,
    collate_fn=batchify_fn)
test_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_size=batch_size,
    return_list=True,
    collate_fn=batchify_fn)

model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

step = 0
for epoch in range(epochs):
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    evaluate(model, metric, dev_loader)

    paddle.save(model.state_dict(),
                './ernie_result/model_%d.pdparams' % step)

state_dict = paddle.load("./ernie_result/model_450.pdparams")
model.load_dict(state_dict)
preds = predict(model, test_loader, test_ds, label_vocab)
file_path = "ernie_results.txt"
with open(file_path, "w", encoding="utf8") as fout:
    fout.write("\n".join(preds))
# Print some examples
print(
    "The results have been saved in the file: %s, some examples are shown below: "
    % file_path)
print("\n".join(preds[:10]))
