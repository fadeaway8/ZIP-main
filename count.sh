#!/bin/sh
python count.py --dataset shanghaitech_a --device cuda:0
python count.py --dataset shanghaitech_b --device cuda:0
python count.py --dataset nwpu           --device cuda:0
python count.py --dataset ucf_qnrf       --device cuda:0
