python data/prepare_data.py --prefix path/to/Deepeyes_Dataset/

# export WANDB_DISABLED=true

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    grpo/train_grpo.py \
    --output_dir "./checkpoints/TiCO/" \
    --model_name_or_path "./checkpoints/TiCO_SFT" \
    --dataset_name "data/tico_grpo.jsonl" \
    --deepspeed scripts/zero3.json \
    --max_prompt_length 16384\
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name stage2_rethink \
    --save_steps 4000 \
    --save_total_limit 10 \
    --max_grad_norm 30 \
    --save_only_model false \
    --beta 0.4 \
    --num_generations_stage1 8 \
    --num_generations 8 
