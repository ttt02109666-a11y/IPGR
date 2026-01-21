# IPGR
Point Cloud Completion Refinement

<img width="1671" height="641" alt="框架图新" src="https://github.com/user-attachments/assets/7fd9dbe0-d613-4622-988b-d7d362767963" />

Insatllation(depend describe)

Quick Start 

test baseline 
python tools/test_ipgr.py --config/xxx.yaml --ckpt path/to/ckpt.pth --no ipgr

test baseline+IPGR
python tools/test_ipgr.py --config/xxx.yaml --ckpt path/to/ckpt.pth --base_alpha 0.05 --num_iter 2
