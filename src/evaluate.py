
import torch
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import argparse
import os

# 导入模型和数据集相关模块
from src.model import Transformer, generate_square_subsequent_mask
from src.dataset import BOS_IDX, EOS_IDX, PAD_IDX, de_tokenizer, parse_xml_file, LOCAL_DATA_PATH

import torchtext
torchtext.disable_torchtext_deprecation_warning()

def greedy_decode(model, src, src_padding_mask, max_len, start_symbol, device):
    #贪心解码函数
    src = src.to(device)
    if src_padding_mask is not None:
        src_padding_mask = src_padding_mask.to(device)
    # 1. 编码输入序列
    memory = model.encode(src, src_padding_mask)
    memory = memory.to(device)
    
    # 2. 初始化目标序列
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device) # shape: (1, 1)

    for i in range(max_len - 1):
        # 3. 创建目标序列的掩码
        tgt_len = ys.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_len, device).unsqueeze(0) # Add batch dimension
        tgt_padding_mask = torch.zeros(ys.shape, device=device).bool() # 推理时没有padding

        # 4. decode
        out = model.decode(ys, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        
        # 5. 选择最后一个时间步概率最高的词元
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 6. 将预测的词元拼接到序列中
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # 7. 如果预测到 EOS，则停止
        if next_word == EOS_IDX:
            break
            
    return ys

def translate(model, src_sentence, de_vocab, en_vocab, device):
    model.eval()
    src_tokens = [BOS_IDX] + de_vocab(de_tokenizer(src_sentence)) + [EOS_IDX]
    src = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    #创建填充掩码
    src_padding_mask = (src == PAD_IDX)
    
    tgt_tokens = greedy_decode(model, src, src_padding_mask, max_len=src.shape[1] + 10, start_symbol=BOS_IDX, device=device).flatten()
    
    return " ".join(en_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").strip()

def calculate_bleu(test_iter, model, de_vocab, en_vocab, device):
    #在测试集上计算 BLEU 分数
    targets = []
    predictions = []
    
    for de_sent, en_sent in tqdm(test_iter, desc="Calculating BLEU"):
        pred_sent = translate(model, de_sent, de_vocab, en_vocab, device)
        predictions.append(pred_sent.split())
        targets.append([en_sent.split()]) 
    # 过滤掉空的预测，防止报错
    predictions = [p if p else [""] for p in predictions]

    return bleu_score(predictions, targets)


def main():
    parser = argparse.ArgumentParser(description='Evaluate the Transformer model')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: Checkpoint未找到{args.checkpoint}")
        return
    #用保存的参数加载
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']
    print("加载模型的参数如下:")
    print(model_args)
    de_vocab = checkpoint['de_vocab']
    en_vocab = checkpoint['en_vocab']
    src_vocab_size = len(de_vocab)
    tgt_vocab_size = len(en_vocab)
    use_pe = not getattr(model_args, 'no_pe', False) and not getattr(model_args, 'use_rpe', False)
    use_residual = not getattr(model_args, 'no_residual', False)
    use_rpe = getattr(model_args, 'use_rpe', False)
    max_len = getattr(model_args, 'max_len', 512) # 默认是512

    model = Transformer(model_args.num_encoder_layers, model_args.num_decoder_layers, 
                        model_args.d_model, model_args.nhead, src_vocab_size, 
                        tgt_vocab_size, model_args.dim_feedforward, model_args.dropout,use_positional_encoding=use_pe,
        use_residual=use_residual,
        use_rpe=use_rpe,           
        max_len=max_len).to(device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[10:]
            new_state_dict[name] = v
        state_dict = new_state_dict                    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功")

    #从本地文件加载测试集
    print("加载测试数据集...")
    # 定义测试集文件路径 
    test_src_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.tst2015.de-en.de.xml')
    test_tgt_path = os.path.join(LOCAL_DATA_PATH, 'IWSLT17.TED.tst2015.de-en.en.xml')

    #解析文件
    test_src_sents = parse_xml_file(test_src_path)
    test_tgt_sents = parse_xml_file(test_tgt_path)
    test_iter = list(zip(test_src_sents, test_tgt_sents))
    print(f"读取{len(test_iter)} 个测试样例")

    #评估并打印
    bleu = calculate_bleu(test_iter, model, de_vocab, en_vocab, device)
    print(f"\nTest BLEU Score: {bleu * 100:.2f}")

    # 示例翻译
    print("\n实例翻译")
    src_sentence = "Ein Mann in einem blauen Hemd spielt Gitarre."
    translation = translate(model, src_sentence, de_vocab, en_vocab, device)
    print(f"Source: {src_sentence}")
    print(f"Translated: {translation}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()