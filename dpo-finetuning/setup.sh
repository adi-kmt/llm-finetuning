pip install -U transformers datasets git+https://github.com/huggingface/peft accelerate trl bitsandbytes
pip install torch==2.1 optimum scipy wandb
git config --global credential.helper store
huggingface-cli login
wandb login