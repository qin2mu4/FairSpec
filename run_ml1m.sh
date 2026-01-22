export CUDA_VISIBLE_DEVICES=6
export WANDB_DISABLED=true
export SWANLAB_DISABLED=true
export SWANLAB_MODE=disabled
BASE_PATH="/home/user/lm_models"
BASE_FILE_PATH="/home/user/code/llm_bias/FairSpec"
OUTPUT="${BASE_FILE_PATH}/output"


MODEL="Llama-3.1-8B-Instruct"
MODEL_FULL_PATH="${BASE_PATH}/${MODEL}"
DATASET="llama3"
python run.py \
  --model_name_or_path "${MODEL_FULL_PATH}" \
  --tokenizer_name_or_path "${MODEL_FULL_PATH}" \
  --model_name "${DATASET}" \
  --dataset_dir "${BASE_FILE_PATH}/data/ml1m/${DATASET}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train True \
  --do_eval True \
  --seed 41 \
  --bf16 True \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --learning_rate 2e-6 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_total_limit 5 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 8 \
  --max_seq_length 1024 \
  --output_dir "${OUTPUT}/ml1m_${DATASET}" \
  --ddp_timeout 500 \
  --logging_first_step True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_nums 8 \
  --blc_alpha 0.0 \
  --blc_weight 0.0 \
  --trainable "gate_proj,down_proj,up_proj" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --load_in_kbits 16 \
  --ddp_find_unused_parameters False \
  --overwrite_output_dir True \
  --load_best_model_at_end False \
  --greater_is_better True \
  --fairness_coef 0.05 \
  --fair_e "0,1,2,3" \
  --sens_size "4,2" \
  &> "${OUTPUT}/log/ml1m_${DATASET}.log"


MODEL="Llama-3.2-3B-Instruct"
MODEL_FULL_PATH="${BASE_PATH}/${MODEL}"
DATASET="llama3_3b"
python run.py \
  --model_name_or_path "${MODEL_FULL_PATH}" \
  --tokenizer_name_or_path "${MODEL_FULL_PATH}" \
  --model_name ${DATASET} \
  --dataset_dir "${BASE_FILE_PATH}/data/ml1m/${DATASET}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train True \
  --do_eval True \
  --seed 41 \
  --bf16 True \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_total_limit 5 \
  --save_steps 500 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 8 \
  --max_seq_length 1024 \
  --output_dir "${OUTPUT}/ml1m_${DATASET}" \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_nums 8 \
  --blc_alpha 0.0 \
  --blc_weight 0.0 \
  --trainable "gate_proj,down_proj,up_proj" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --load_in_kbits 16 \
  --ddp_find_unused_parameters False \
  --overwrite_output_dir True \
  --load_best_model_at_end False \
  --greater_is_better True \
  --fairness_coef 0.05 \
  --fair_e "0,1,2,3" \
  --sens_size "4,2" \
  &> "${OUTPUT}/log/ml1m_${DATASET}.log"


MODEL="Llama2_7b_chat_hf"
MODEL_FULL_PATH="${BASE_PATH}/${MODEL}"
DATASET="llama2"
python run.py \
  --model_name_or_path "${MODEL_FULL_PATH}" \
  --tokenizer_name_or_path "${MODEL_FULL_PATH}" \
  --model_name ${DATASET} \
  --dataset_dir "${BASE_FILE_PATH}/data/ml1m/${DATASET}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train True \
  --do_eval True \
  --seed 41 \
  --bf16 True \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_total_limit 5 \
  --save_steps 500 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 8 \
  --max_seq_length 1024 \
  --output_dir "${OUTPUT}/ml1m_${DATASET}" \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_nums 8 \
  --blc_alpha 0.0 \
  --blc_weight 0.0 \
  --trainable "gate_proj,down_proj,up_proj" \
  --lora_dropout 0.1 \
  --torch_dtype bfloat16 \
  --load_in_kbits 16 \
  --ddp_find_unused_parameters False \
  --overwrite_output_dir True \
  --load_best_model_at_end False \
  --greater_is_better True \
  --fairness_coef 0.05 \
  --fair_e "0,1,2,3,4,5" \
  --sens_size "4,2" \
  &> "${OUTPUT}/log/ml1m_${DATASET}.log"


MODEL="Qwen2.5-7B-Instruct"
MODEL_FULL_PATH="${BASE_PATH}/${MODEL}"
DATASET="qwen25"
python run.py \
  --model_name_or_path "${MODEL_FULL_PATH}" \
  --tokenizer_name_or_path "${MODEL_FULL_PATH}" \
  --model_name ${DATASET} \
  --dataset_dir "${BASE_FILE_PATH}/data/ml1m/${DATASET}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train True \
  --do_eval True \
  --seed 41 \
  --bf16 True \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy steps \
  --save_total_limit 5 \
  --save_steps 500 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 8 \
  --max_seq_length 1024 \
  --output_dir "${OUTPUT}/ml1m_${DATASET}" \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_nums 8 \
  --blc_alpha 0.0 \
  --blc_weight 0.0 \
  --trainable "gate_proj,down_proj,up_proj" \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --load_in_kbits 16 \
  --ddp_find_unused_parameters False \
  --overwrite_output_dir True \
  --load_best_model_at_end False \
  --greater_is_better True \
  --fairness_coef 0.05 \
  --fair_e "0,1,2,3" \
  --sens_size "4,2" \
  &> "${OUTPUT}/log/ml1m_${DATASET}.log"
  
  


