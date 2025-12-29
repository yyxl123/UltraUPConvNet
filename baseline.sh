export CUDA_VISIBLE_DEVICES=0

python train.py --output_dir=exp_out/prompt     --prompt     --base_lr=3e-5 --use_pretrained --batch_size=16

#reume
python train.py --output_dir=exp_out/trial_1     --prompt     --base_lr=0.003 --use_pretrained --batch_size=16 --resume=/MICCAI/ours/exp_out/trial_1/latest.pth

python  test.py     --output_dir=exp_out/trial_1     --prompt     --is_saveout