export CUDA_VISIBLE_DEVICES=0

python train.py --output_dir=exp_out/prompt     --prompt     --base_lr=3e-5 --use_pretrained --batch_size=16

python  test.py     --output_dir=exp_out/prompt     --prompt     --is_saveout