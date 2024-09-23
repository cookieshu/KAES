#!/usr/bin/env bash
model_name='KAES'
for seed in 9 12 42 51 86
do
    for prompt in {1..6}
    do
        python train_KAES.py --test_prompt_id ${prompt} --model_name ${model_name} --seed ${seed} --num_heads 2 --features_path 'data/hand_crafted_feature.csv'
    done
done