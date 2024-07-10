import yaml
import argparse
from diffusion import Diffusion
import pytorch_lightning as pl
# from ema import EMA, EMAModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset
from data import get_dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import click
import os
import datetime
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint


@rank_zero_only
def create_directories(path):
    '''
    Makes sure to create directoris for only process 0 in multi-GPU scenarios
    '''
    os.makedirs(path, exist_ok=True)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

@click.command()
@click.option('--cfg', default='config.yml', help='Configuration File')
def main(cfg):
    config = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = dict2namespace(cfg)

    log_path = cfg.log_directory
    create_directories(log_path)

    # adding a random number of seconds so that exp folder names coincide less often
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    nowname = "%s_%s_%s" % (
        now,
        cfg.id.name,
        cfg.id.version,
        # int(time.time())
    )

    print("\nName of the run is:", nowname, "\n")

    run_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
    )

    create_directories(run_path)

    wandb_logger = WandbLogger(
        save_dir=run_path,
        # version=nowname,
        project= cfg.project_name,
        config=config,
        name=nowname
    )
    wandb_logger._project = ""  # prevent naming experiment nama 2 time in logginf vals

    checkpoint_path = os.path.join(
        log_path,
        cfg.project_name,
        nowname,
        "checkpoints",
    )
    create_directories(checkpoint_path)

    # ckpt_callback = EMAModelCheckpoint(dirpath= checkpoint_path, save_top_k=cfg.training.save_top_k, monitor="val_loss", save_last=True, filename='{epoch}-{val_loss:.4f}', every_n_train_steps=None,)
    # ema_callback = EMA(decay=cfg.model.ema_rate)
    # callbacks = [ckpt_callback, ema_callback]

    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=cfg.training.save_top_k,
        monitor="val_loss",
        save_last=True,
        filename='{epoch}-{val_loss:.4f}',
        every_n_train_steps=None
    )

    callbacks = [ckpt_callback]

    model = Diffusion(cfg)

    train_dataset = get_dataset(cfg.data.name, train=True)
    val_dataset = get_dataset(cfg.data.name, train=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.testing.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )

    # Check if train_dataloader.dataset has the 'classes' attribute and add it to model.cfg if it exists
    if hasattr(train_dataloader.dataset, 'classes'):
        model.cfg.class_names = train_dataloader.dataset.classes
    else:
        model.cfg.class_names = None  # Optionally set to None if classes do not exist

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        precision=cfg.training.precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        accelerator="gpu", 
        devices=cfg.training.devices,
        num_sanity_val_steps=0,
        limit_val_batches=10,
        limit_train_batches=10,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.validation_every_n_epochs,
        val_check_interval=cfg.training.val_check_interval, 
        gradient_clip_val=cfg.optim.grad_clip,
        benchmark=True,
        strategy = DDPStrategy(find_unused_parameters=False)
                    if isinstance(cfg.training.devices, (list, tuple)) and len(cfg.training.devices) > 1
                    else "auto",  # Ensure cfg.training.devices is a list or tuple
        # DDPStrategy(find_unused_parameters=False),
    )

    # Train
    # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)
    # trainer.validate(model, val_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)

    if cfg.mode in ["validate_all"]:
        # Evaluation / Validation
        
        # Create a combined dataset
        combined_dataset = ConcatDataset([train_dataset, val_dataset])

        # Create a DataLoader for the combined dataset
        combined_dataloader = DataLoader(
            combined_dataset, 
            batch_size=cfg.testing.batch_size, 
            shuffle=False, 
            num_workers=cfg.data.num_workers, 
            pin_memory=True, 
            persistent_workers=True,
        )

        trainer.validate(model, combined_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)
    elif cfg.mode in ["test", "validate"]:
        # Evaluation / Validation
        trainer.validate(model, val_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)
    elif cfg.mode == "train":
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path = cfg.training.resume_from_checkpoint)



if __name__ == '__main__':
    main()