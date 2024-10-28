
export CUDA_VISIBLE_DEVICES=2

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

for learning_rate in 3e-5
do
for data in Gas
do
for subset_rand_ratio in 1
do
i=1
for ckpt_path in checkpoints/Timer_forecast_1.0.ckpt
do
python -u run.py \
  --task_name large_finetune \
  --is_training 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/Gas/ \
  --data_path 134312_data.csv \
  --data $data \
  --model_id 2G_{$seq_len}_{$pred_len}_{$patch_len}_ \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 32 \
  --finetune_epochs 50 \
  --learning_rate $learning_rate \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 1 \
  --subset_rand_ratio $subset_rand_ratio \
  --train_offset $patch_len \
  --itr 1 \
  --gpu 0 \
  --roll \
  --show_demo \
  --is_finetuning 0
i=$((i+2))
done
done
done
done