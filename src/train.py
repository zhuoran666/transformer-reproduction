
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import argparse
import time
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from src.model import Transformer, create_mask
from src.dataset import get_dataloaders, PAD_IDX

def get_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train the Transformer model')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_name', type=str, default=f'transformer_{time.strftime("%Y%m%d-%H%M%S")}')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--no_pe', action='store_true',help='禁用绝对位置编码')
    parser.add_argument('--no_residual', action='store_true',help='禁用残差连接')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_rpe', action='store_true',help='使用相对位置编码')
    parser.add_argument('--max_len', type=int, default=512,help='最大序列长度')

    return parser.parse_args()

def get_lr_scheduler(optimizer, warmup_steps):
# 学习率调度器，包含预热和反向平方根衰减
    def lr_lambda(current_step: int):
        current_step += 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return float(warmup_steps ** 0.5) * float(current_step ** -0.5)
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, optimizer, scheduler, criterion, train_dataloader, device, clip_norm, log_interval, epoch, total_epochs, scaler):
    model.train()
    losses = 0
    num_batches = len(train_dataloader)
#一个训练轮次
    for i,(src, tgt) in enumerate(train_dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        # 混合精度AMP训练
        with autocast():
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,src_padding_mask)
            tgt_out = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        # 反向传播和优化步骤
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        losses += loss.item()

        # 日志打印
        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch}/{total_epochs}] | Step [{i + 1}/{num_batches}] | Loss: {loss.item():.4f} | LR: {lr:.6e}",flush=True)

    return losses / num_batches

def evaluate(model, criterion, val_dataloader, device):
    #验证性能
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Validating"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
            
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[:,1 :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            
    return losses / len(list(val_dataloader))

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    # 路径设置
    save_dir = os.path.join("results", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载
    train_loader, val_loader, _, de_vocab, en_vocab = get_dataloaders(args.batch_size)
    src_vocab_size = len(de_vocab)
    tgt_vocab_size = len(en_vocab)
    print(f"SRC Vocab Size: {src_vocab_size}, TGT Vocab Size: {tgt_vocab_size}")
    scaler = GradScaler(enabled=torch.cuda.is_available())
    use_pe = not args.no_pe and not args.use_rpe
    # 模型初始化
    model = Transformer(args.num_encoder_layers, args.num_decoder_layers, args.d_model,
                        args.nhead, src_vocab_size, tgt_vocab_size, args.dim_feedforward,
                        args.dropout,use_positional_encoding=use_pe,use_residual=not args.no_residual,
                        use_rpe=args.use_rpe,
                        max_len=args.max_len).to(device)
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    try:
        model = torch.compile(model)
        print("模型编译完成")
    except Exception as e:
        print(f"模型编译失败: {e}")
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型含有{total_params:,} 可训练参数")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, scheduler, criterion, train_loader, device, 
                                 args.clip_grad_norm, args.log_interval, epoch, args.epochs, scaler)
        end_time = time.time()
        val_loss = evaluate(model, criterion, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
              f"Epoch time: {(end_time - start_time):.3f}s")
        print(f"Train PPL: {math.exp(train_loss):.3f}, Val PPL: {math.exp(val_loss):.3f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(), # 使用 model_to_save
                'de_vocab': de_vocab,
                'en_vocab': en_vocab,
                'args': args
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"最佳模型已保存至{save_dir}/best_model.pt")

    # 绘制并保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print(f"训练曲线保存至{save_dir}/loss_curve.png")

if __name__ == "__main__":
    main()