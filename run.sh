#!/usr/bin/env bash
python main.py --data_dir data/ --model_type bert --model_name_or_path bert-base-chinese \
--task_name MyProcessor --output_dir out/ --cache_dir cache --max_seq_length 300 --do_train \
--evaluate_during_training --do_lower_case --overwrite_output_dir #--eval_all_checkpoints
