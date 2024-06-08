import datetime

def create_run_info(config):
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name += f'[{config.wandb.run_name_suffix}]' if config.wandb.run_name_suffix else ''
    
    model_cfg = config.model.cfg
    audio_info = f'{model_cfg.n_in_channel}chan_{model_cfg.sr}Hz_{model_cfg.clip_len}s_{model_cfg.n_clip_segment}segs'
    
    return run_name, audio_info