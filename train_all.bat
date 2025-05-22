

REM training for 8000 steps, 35 chunk size
python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir "./temp_8000_steps_35" --policy_class ACT --kl_weight 10 --chunk_size 35 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 8000 --lr 1e-5 --seed 0

REM training for 8000 steps, 20 chunk size
python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir "./temp_8000_steps_20" --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 8000 --lr 1e-5 --seed 0

REM training for 8000 steps, 10 chunk size
python imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir "./temp_8000_steps_10" --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 8000 --lr 1e-5 --seed 0

REM finished!
echo All tasks have been executed!
pause
