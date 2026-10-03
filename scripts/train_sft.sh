
model_type="qwen25vl"
llm_model="path/to/Qwen2.5-VL-7B-Instruct"
output_dir="./checkpoints/TiCO_SFT"
# export WANDB_DISABLED=true

# === Training Command ===
torchrun --nproc_per_node=4 sft/tico/train_sft.py \
  --deepspeed ./scripts/zero3.json \
  --data_path data/sft_data_config.yaml \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "epoch" \
  --save_steps 640 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels 2000000 \
  --unfreeze_all_parameters True \
  --unfreeze_pointer_head True \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_new_tokens True \
  --unfreeze_visual True \
  --max_grad_norm 30 \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 1.0
