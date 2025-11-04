export WANDB_PROJECT=rnd-sft
export WANDB_ENTITY=stanford_autonomous_agent    # change to RN once have access

accelerate launch \
    --config_file scripts/accelerate_configs/zero3.yaml \
    examples/rnd/sft_v3.py
    # examples/rnd/sft.py