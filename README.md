# IPGR
Point Cloud Completion Refinement

Insatllation(depend describe)

Quick Start 

test baseline 
python tools/test_ipgr.py --config/xxx.yaml --ckpt path/to/ckpt.pth --no ipgr

test baseline+IPGR
python tools/test_ipgr.py --config/xxx.yaml --ckpt path/to/ckpt.pth --base_alpha 0.05 --num_iter 2
