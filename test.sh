# # ShanghaiTech A
# python test.py --dataset sha --weight_path checkpoints/sha/ebc_p_best/best_mae.pth --output_filename ebc_p_best_mae --amp 

# python test.py --dataset sha --weight_path checkpoints/sha/ebc_n_best/best_mae.pth --output_filename ebc_n_best_mae --amp 

# python test.py --dataset sha --weight_path checkpoints/sha/ebc_t_best/best_mae.pth --output_filename ebc_t_best_mae --amp 

# python test.py --dataset sha --weight_path checkpoints/sha/ebc_s_best/best_mae.pth --output_filename ebc_s_best_mae --amp 

# python test.py --dataset sha --weight_path checkpoints/sha/ebc_b_vit_best/best_mae.pth --output_filename ebc_b_best_mae --sliding_window --input_size 224 --amp


# # ShanghaiTech B
# python test.py --dataset shb --weight_path checkpoints/shb/ebc_p_best/best_mae.pth --output_filename ebc_p_best_mae --amp

# python test.py --dataset shb --weight_path checkpoints/shb/ebc_n_best/best_mae.pth --output_filename ebc_n_best_mae --amp

# python test.py --dataset shb --weight_path checkpoints/shb/ebc_t_best/best_mae.pth --output_filename ebc_t_best_mae --amp 

# python test.py --dataset shb --weight_path checkpoints/shb/ebc_s_best/best_mae.pth --output_filename ebc_s_best_mae --amp

# python test.py --dataset shb --weight_path checkpoints/shb/ebc_b_best/best_mae.pth --output_filename ebc_b_best_mae --amp 


# # UCF-QNRF
# python test.py --dataset qnrf --weight_path checkpoints/qnrf/ebc_p_best/best_mae.pth --input_size 672 --output_filename ebc_p_best_mae --amp

# python test.py --dataset qnrf --weight_path checkpoints/qnrf/ebc_n_best/best_mae.pth --input_size 672 --output_filename ebc_n_best_mae --amp

# python test.py --dataset qnrf --weight_path checkpoints/qnrf/ebc_t_best/best_mae.pth --input_size 672 --output_filename ebc_t_best_mae --amp

# python test.py --dataset qnrf --weight_path checkpoints/qnrf/ebc_s_best/best_mae.pth --input_size 672 --output_filename ebc_s_best_mae --amp

# python test.py --dataset qnrf --weight_path checkpoints/qnrf/ebc_b_best/best_mae.pth --input_size 672 --output_filename ebc_b_best_mae --amp


# NWPU
python test.py --dataset nwpu --weight_path checkpoints/nwpu/ebc_b_best/best_mae.pth --input_size 672 --output_filename ebc_b_best_mae --amp --device mps