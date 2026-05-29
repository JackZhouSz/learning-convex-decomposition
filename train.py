# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from model.config import default_argument_parser, setup
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import glob
import os
import sys
import shutil


class OutputCapture:
    """Capture stdout and stderr to file while preserving original output"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        
    def __enter__(self):
        # Only capture output on main process
        if is_main_process():
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self.log_file_handle = open(self.log_file, 'w', buffering=8192)  # Use buffered writes
            sys.stdout = self.TeeOutput(self.terminal_stdout, self.log_file_handle)
            sys.stderr = self.TeeOutput(self.terminal_stderr, self.log_file_handle)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_main_process():
            sys.stdout = self.terminal_stdout
            sys.stderr = self.terminal_stderr
            self.log_file_handle.close()
    
    class TeeOutput:
        def __init__(self, terminal, file_handle):
            self.terminal = terminal
            self.file_handle = file_handle
            
        def write(self, message):
            self.terminal.write(message)
            self.file_handle.write(message)
            # Only flush on newlines or when buffer is full (less frequent)
            if '\n' in message:
                self.file_handle.flush()
            
        def flush(self):
            self.terminal.flush()
            self.file_handle.flush()


def is_main_process():
    """Check if this is the main process in DDP"""
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def collect_ckpts(cfg):
    """
    Return a sorted list of checkpoint paths matching
      • .../epoch=*.ckpt
      • .../last.ckpt
    under the results/<output_dir‑parent>/* hierarchy.
    """
    current_dir = os.getcwd()
    base = os.path.join(current_dir, "exp_results", cfg.output_dir.split('/')[-2], "*")

    ckpt_paths = []
    for patt in ("epoch=*.ckpt", "last.ckpt"):
        ckpt_paths += glob.glob(os.path.join(base, patt))

    # alphabetical sort keeps epoch order; adjust if you need mtime sort
    return sorted(ckpt_paths)

def train(cfg):
    seed_everything(cfg.seed)
    
    # Setup output capture for console logs
    log_file = os.path.join(cfg.output_dir, "training_output.log")
    

    with OutputCapture(log_file):

        from model.model_trainer import Model
        model = Model(cfg)        

        checkpoint_callbacks = [ModelCheckpoint( dirpath=cfg.output_dir, filename="{epoch:02d}", save_top_k=-1,
            save_last=False, every_n_epochs=cfg.save_every_epoch, mode="max",verbose=True),
            ModelCheckpoint( dirpath=cfg.output_dir, filename=None,  save_top_k=0, save_last=True, every_n_train_steps=250, save_on_train_epoch_end=False  ) ]



        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          precision="16-mixed",
                          strategy=DDPStrategy(find_unused_parameters=True),
                          max_epochs=cfg.training_epochs,
                          log_every_n_steps=1,
                          limit_train_batches=cfg.limit_train_batches,
                          limit_val_batches=None,
                          profiler="simple",
                          callbacks= checkpoint_callbacks
                         )
        

        ckpt_paths = collect_ckpts(cfg) 


        if cfg.continue_training and (len(ckpt_paths) != 0 or cfg.ckpt_path is not None):
            if len(ckpt_paths) != 0:
                for i in range(1, len(ckpt_paths) + 1):
                    ckpt_path = ckpt_paths[-i]
                    if os.path.exists(ckpt_path):
                        break
                else:
                    ckpt_path = ckpt_paths[-1]
            else:
                ckpt_path = cfg.ckpt_path

            if is_main_process():
                print("loading latest ckpt: ", ckpt_path)


            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)
        
def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)
    train(cfg)
    
if __name__ == '__main__':
    main()
