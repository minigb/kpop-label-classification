from PIL import Image
from pathlib import Path
import csv
import datetime
import wandb
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

import model_zoo

  
def nll_loss(pred, target):
  loss = -torch.log(pred[range(len(target)), target]) 
  loss = loss.mean()
  return loss
  
def custom_collate_fn(batch): # RuntimeError: Trying to resize storage that is not resizable
  batch_items = list(zip(*batch))

  if len(batch_items) == 2:
    audio_tensors, labels = zip(*batch) 
    audio_tensors_padded = pad_sequence(audio_tensors, batch_first=True)
    labels = torch.tensor(labels)
    return audio_tensors_padded, labels
  
  elif len(batch_items) == 3:
    audio_tensors, labels, file_names = zip(*batch)
    audio_tensors_padded = pad_sequence(audio_tensors, batch_first=True)
    labels = torch.tensor(labels)
    return audio_tensors_padded, labels, file_names
  

@hydra.main(config_path='config', config_name='packed')
def main(config):
  DEV = 'cuda'

  run_name = f'{datetime.datetime.now().strftime("%m%d_%H%M")}'
  wandb.init(project='kpop-label-classification', name=run_name)
  wandb.config.update(OmegaConf.to_container(config))
  
  model_class = getattr(model_zoo, config.model.cls)
  model = model_class(**config.model.cfg).to(DEV)
  model = nn.DataParallel(model) if torch.cuda.device_count() > 1 and config.model.cls != 'CRNN' else model.to(DEV)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
  collate_func = custom_collate_fn
  
  save_dir = Path(f'weights/{datetime.datetime.now().strftime("%m%d_%H%M")}_{config.model.cls}')
  save_dir.mkdir(parents=True, exist_ok=True)
  with open(save_dir/'config.yaml', 'w') as f:
    OmegaConf.save(config, f)

  best_model_dir = Path(f'best_model/{save_dir.name}')
  best_model_dir.mkdir(parents=True, exist_ok=True)
  with open(best_model_dir/'config.yaml', 'w') as f:
    OmegaConf.save(config, f)
    
  validset = BillboardDatasetHierarchyValidTest(**config.valid_test, 
                                                mode='valid')
  validation_loader = DataLoader(validset, 
                                batch_size=config.train.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True,
                                collate_fn=custom_collate_fn)
  if config.train.adversarial_loss_weight > 0:    
    adv_set = BillboardDatasetHierarchyTrain(**config.adv_set, mode='train', chunk_indices=None)
    adv_loader = DataLoader(adv_set, 
                            batch_size=config.train.batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True,
                            collate_fn=custom_collate_fn)
  else:
    adv_loader = None
  
  testset = BillboardDatasetHierarchyValidTest(**config.valid_test,
                                              mode='test')
  test_loader = DataLoader(testset, 
                          batch_size=config.train.batch_size, 
                          shuffle=False, 
                          num_workers=4, 
                          pin_memory=True, 
                          collate_fn=collate_func)
  trainer = Trainer(model, 
                    optimizer, 
                    nll_loss, 
                    validation_loader, 
                    config.train.iterations_per_chunk, 
                    DEV, 
                    config.train.validation_freq,
                    collate_fn=custom_collate_fn,
                    save_dir=save_dir,
                    best_model_dir=best_model_dir,
                    hierarchy_class_map=validset.hierarchy_class_map,
                    high_to_low_weight=config.train.high_to_low_weight,
                    low_to_high_weight=config.train.low_to_high_weight,
                    config=config,
                    adv_loader=adv_loader,
                    test_loader=test_loader,)
  
  
  trainer.train_model_with_chunks(**config.train_model_with_chunks) 
  
  # trainer.load_best_model() # todo
  # trainer.get_test_results()
  
  # test on kpop dataset
  # inferencer = Inference()
  # inferencer.test_model_on_kpop() # todo

if __name__ == '__main__':
  main()