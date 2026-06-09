import torch
import yaml
import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.datasets.dataset import EpicKitchensFeatureModule, EpicKitchensFramesModule
from src.models.adapter import AdapterModule
from src.models.clip import VideoCLIPModule
from transformers import AutoTokenizer

import transformers
import tokenizers
torch.serialization.add_safe_globals([transformers.models.bert.tokenization_bert.BertTokenizer])
torch.serialization.add_safe_globals([tokenizers.Tokenizer])
torch.serialization.add_safe_globals([tokenizers.models.Model])
torch.serialization.add_safe_globals([tokenizers.AddedToken])

class ModelTrainer():
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        train_config = config['train_config']

        model_name = config['model_config']['model']

        if model_name == "adapter":
            self.model = AdapterModule(
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                adapter_config=config['model_config'],
                loss=train_config['loss']
            )

            self.module = EpicKitchensFeatureModule(
                csv_dir=train_config['csv_dir'],
                features_dir=train_config['feature_dir'],
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
            )

        elif model_name == "clip":
            self.model = VideoCLIPModule(
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                adapter_config=config['model_config']
            )

            distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            self.module = EpicKitchensFramesModule(
                csv_dir=train_config['csv_dir'],
                frames_dir = train_config['frames_dir'],
                tokenizer = distilbert_tokenizer,
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        logging_config = config['logging_config']

        self.logger = WandbLogger(
            project=logging_config['project_name'], 
            name = logging_config['exp_name'],
            save_dir = logging_config['log_dir']
        )

        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor='val/loss', patience=train_config['patience'], mode='min', verbose=True),
            ModelCheckpoint(
                dirpath = logging_config['checkpoint_dir'],
                monitor='val/loss',
                filename = logging_config['exp_name']+'{epoch}-{val/loss:.2f}',
                mode='min', 
                save_last=True
            )
        ]

        self.trainer = Trainer(
            accelerator = train_config['accelerator'],
            max_epochs = train_config['max_epochs'],
            logger = self.logger,
            callbacks = callbacks,
            log_every_n_steps = 50,
            accumulate_grad_batches = train_config['batch_acc'],
            precision = "bf16-mixed",
        )

    def train(self, ckpt_path=None):
        self.module.setup('fit')
        self.trainer.fit(self.model, datamodule=self.module, ckpt_path=ckpt_path)

    def test(self, ckpt_path=None):
        self.module.setup('test')
        self.trainer.test(self.model, datamodule=self.module, ckpt_path=ckpt_path)

    def validate(self, ckpt_path=None):
        self.module.setup('fit')
        self.trainer.validate(self.model, datamodule=self.module, ckpt_path=ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--test', action='store_true', help='only test mode')
    parser.add_argument('--ckpt', type=str, default=None, help='restore weights from checkpoint')

    user_args = parser.parse_args()

    trainer = ModelTrainer(user_args.config)
    
    if user_args.test:
        trainer.test(user_args.ckpt)
    else:
        trainer.train(user_args.ckpt)
