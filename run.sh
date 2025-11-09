
# 实验结构:
#   - baseline: 一个经过验证的3层模型，作为新的性能基准。
#   - ablations:    围绕新基线进行的一系列消融实验。
#   - challenge:    尝试通过强正则化来优化原始的6层大模型。
# 使用方法:
#   ./run.sh <experiment_name>
# -----------------------------------------------------------------------------

set -e

# --- 实验执行函数---
run_experiment() {
    local run_name=$1
    local epochs=$2
    local d_model=$3
    local num_layers=$4
    local nhead=$5
    local d_ff=$6
    local dropout=$7
    local label_smoothing=$8
    local lr=${9}
    local weight_decay=${10}
    local extra_train_args=${11}

    echo ""
    echo "====================================================================="
    echo "  开始实验: ${run_name}"
    echo "====================================================================="
    echo "超参数:"
    echo "  - Epochs: ${epochs}, Layers: ${num_layers}"
    echo "  - d_model: ${d_model}, d_ff: ${d_ff}, Heads: ${nhead}"
    echo "  - Dropout: ${dropout}, Label Smoothing: ${label_smoothing}"
    echo "  - Learning Rate: ${lr}, Weight Decay: ${weight_decay}"
    if [ -n "$extra_train_args" ]; then
        echo "  - Extra Args: ${extra_train_args}"
    fi
    echo "---------------------------------------------------------------------"

    local RUN_DIR="results/${run_name}"
    mkdir -p "$RUN_DIR"
    local LOG_FILE="$RUN_DIR/log.txt"

    {
        python -u -m src.train \
            --epochs "$epochs" \
            --batch_size 64 \
            --d_model "$d_model" \
            --nhead "$nhead" \
            --num_encoder_layers "$num_layers" \
            --num_decoder_layers "$num_layers" \
            --dim_feedforward "$d_ff" \
            --dropout "$dropout" \
            --lr "$lr" \
            --warmup_steps 4000 \
            --clip_grad_norm 1.0 \
            --seed 42 \
            --run_name "$run_name" \
            --log_interval 200 \
            --label_smoothing "$label_smoothing" \
            --weight_decay "$weight_decay" \
            $extra_train_args

        echo -e "\n--- 验证阶段 ---"
        local BEST_MODEL_PATH="$RUN_DIR/best_model.pt"
        if [ -f "$BEST_MODEL_PATH" ]; then
            python3 -u -m src.evaluate --checkpoint "$BEST_MODEL_PATH"
        else
            echo "错误:未在该路径找到best_model.pt: $BEST_MODEL_PATH"; exit 1;
        fi
    } 2>&1 | tee "$LOG_FILE"

    echo "====================================================================="
    echo "  实验结束: ${run_name}"
    echo "====================================================================="
    echo ""
}

EXPERIMENT=$1

case $EXPERIMENT in
    # --- 1. 新基线 ---
    baseline)
        # 基于之前的训练结果，配置更长的训练时间以充分收敛
        run_experiment \
            "baseline_d512_L3_H8" \
            20 \
            512 \
            3 \
            8 \
            2048 \
            0.1 \
            0.1 \
            0.0005 \
            0.0 \
            ""
        ;;
    ablation-heads)
        # 消融实验：与基线相比，将注意力头数从8减少到4
        run_experiment \
            "ablation_heads_d512_L3_H4_ls0.1" \
            20 \
            512 \
            3 \
            4 \
            2048 \
            0.1 \
            0.1 \
            0.0005\
            0.0
        ;;
    ablation-narrower)
        # 消融实验：与基线对比：一个更窄、更快的模型
        run_experiment \
            "ablation_narrower_d256_L3_H8" \
            20 \
            256 \
            3 \
            8 \
            1024 \
            0.1 \
            0.1 \
            0.0005 \
            0.0 \
            ""
        ;;
    ablation-narrower-regularized)
        # 消融实验：在 'narrower' 模型的基础上，增加了 Dropout 到 0.3，引入了 Weight Decay (0.01)
        run_experiment \
            "ablation_narrower-regularized" \
            20 \
            256 \
            3 \
            8 \
            1024 \
            0.3 \
            0.1 \
            0.0005 \
            0.01 \
            ""
        ;;
    ablation-more-dropout)
        # 消融实验：增加 Dropout
        run_experiment \
            "ablation_more-dropout_d512_L3_H8" \
            20 \
            512 \
            3 \
            8 \
            2048 \
            0.3 \
            0.1 \
            0.0005 \
            0.0 \
            ""
        ;;
    ablation-no-ls-v2)
        # 消融实验：移除标签平滑
        run_experiment \
            "ablation_no-ls-v2_d512_L3_H8" \
            20 \
            512 \
            3 \
            8 \
            2048 \
            0.1 \
            0.0 \
            0.0005 \
            0.0 \
            ""
        ;;

    challenge-deep-regularized)
        # 消融实验：6层的更大的模型，但加入了更强的正则化（高Dropout和Weight Decay）
        run_experiment \
            "challenge_deep-regularized_d512_L6_H8" \
            20 \
            512 \
            6 \
            8 \
            2048 \
            0.3 \
            0.1 \
            0.0005 \
            0.01 \
            ""
        ;;
    
    ablation-no-pe)
        # 消融实验：移除位置编码
        run_experiment \
            "ablation_no-pe_d512_L3_H8" \
            20 \
            512 \
            3 \
            8 \
            2048 \
            0.1 \
            0.1 \
            0.0005 \
            0.0 \
            "--no_pe"
        ;;
    ablation-ffn-small)
        # 消融实验：将 FFN 的中间维度缩小到与 d_model 相同
        run_experiment \
            "ablation_ffn-small_d512_L6_H8_ls0.1" \
            20 512 3 8 512 0.1 0.1 0.0005 0.0"" 
        ;;
    ablation-shallow)
        # 消融实验：与基线相比，2层模型
        run_experiment \
            "ablation_shallow_d256_L2_H8_ls0.1" \
            20 \
            512 \
            2 \
            8 \
            2048 \
            0.1 \
            0.1 \
            0.0005\
            0.0
        ;;
    ablation-rpe)
        # 消融实验: 相对位置编码 (RPE)
        run_experiment \
            "ablation_rpe_d512_L3_H8" \
            20 \
            512 \
            3 \
            8 \
            2048 \
            0.1 \
            0.1 \
            0.0005 \
            0.0 \
            "--use_rpe"
        ;;
    all)
        echo "运行所有实验..."
        bash "$0" baseline
        bash "$0" ablation-heads
        bash "$0" ablation-narrower
        bash "$0" ablation-narrower-regularized
        bash "$0" ablation-more-dropout
        bash "$0" ablation-no-ls-v2
        bash "$0" challenge-deep-regularized
        bash "$0" ablation-no-pe
        bash "$0" ablation-ffn-small
        bash "$0" ablation-shallow
        bash "$0" ablation-rpe 
        echo "所有实验结束"
        ;;

    *)
        echo "Usage: $0 {baseline|ablation-heads|ablation-narrower|ablation-narrower-regularized|ablation-more-dropout|ablation-no-ls-v2|challenge-deep-regularized|ablation-no-pe|ablation-ffn-small|ablation-shallow|ablation-rpe |all}"
        exit 1
        ;;
esac