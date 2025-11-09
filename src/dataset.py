
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import os
import xml.etree.ElementTree as ET

#数据集本地路径
LOCAL_DATA_PATH = './data/iwslt2017_de_en'

#特殊符号和索引
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# 分词器
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')



#读取文件函数
def read_plain_text_file(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def parse_xml_file(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")
    tree = ET.parse(filepath)
    root = tree.getroot()
    sentences = []
    for seg in root.findall('.//seg'):
        if seg.text:
            sentences.append(seg.text.strip())
    return sentences

# --- Main Data Loading Function (Modified) ---

def get_dataloaders(batch_size):
    train_src_path = os.path.join(LOCAL_DATA_PATH, 'train.tags.de-en.de')
    train_tgt_path = os.path.join(LOCAL_DATA_PATH, 'train.tags.de-en.en')
    #验证和测试集使用XML格式，可选择年份
    val_src_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.dev2010.de-en.de.xml')
    val_tgt_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.dev2010.de-en.en.xml')
    
    test_src_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.tst2015.de-en.de.xml')
    test_tgt_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.tst2015.de-en.en.xml')

    print("正在从本地加载数据...")
    train_src_sents = read_plain_text_file(train_src_path)
    train_tgt_sents = read_plain_text_file(train_tgt_path)
    val_src_sents = parse_xml_file(val_src_path)
    val_tgt_sents = parse_xml_file(val_tgt_path)
    test_src_sents = parse_xml_file(test_src_path)
    test_tgt_sents = parse_xml_file(test_tgt_path)
    train_iter = list(zip(train_src_sents, train_tgt_sents))
    val_iter = list(zip(val_src_sents, val_tgt_sents))
    test_iter = list(zip(test_src_sents, test_tgt_sents))
    print(f"已加载 {len(train_iter)} 训练, {len(val_iter)} 验证, {len(test_iter)} 测试样本")

    
    print("构建词汇表...")
    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter, de_tokenizer, 0),
                                         min_freq=2,
                                         specials=special_symbols,
                                         special_first=True)
    de_vocab.set_default_index(UNK_IDX)

    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter, en_tokenizer, 1),
                                         min_freq=2,
                                         specials=special_symbols,
                                         special_first=True)
    en_vocab.set_default_index(UNK_IDX)
    print("词汇表以构建")

    print("创建数据加载器...")
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, de_vocab, en_vocab))
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, de_vocab, en_vocab))
    test_dataloader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, de_vocab, en_vocab))

    return train_dataloader, val_dataloader, test_dataloader, de_vocab, en_vocab


def yield_tokens(data_iter: Iterable, tokenizer, index: int) -> List[str]:
    for data_sample in data_iter:
        yield tokenizer(data_sample[index])

def collate_fn(batch, de_vocab, en_vocab):
   
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor([BOS_IDX] + de_vocab(de_tokenizer(src_sample)) + [EOS_IDX], dtype=torch.long))
        tgt_batch.append(torch.tensor([BOS_IDX] + en_vocab(en_tokenizer(tgt_sample)) + [EOS_IDX], dtype=torch.long))

    # 填充序列
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_batch, tgt_batch