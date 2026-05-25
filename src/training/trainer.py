import torch
import yaml
import sys
import os

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

class ModelTrainer():
    def __init__(self, config_file_path, model="adapter"):
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        train_config = config['train_config']
        if model == "adapter":
            self.model = AdapterModule(
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                adapter_config=config['model_config']
            )
        elif model == "clip":
            self.model = VideoCLIPModule(
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                adapter_config=config['model_config']
            )
        else:
            raise ValueError(f"Unknown model: {model}")

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
            log_every_n_steps = 50
        )

        if model == "adapter":
            self.module = EpicKitchensFeatureModule(
                features_dir='./data/features',
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
            )
        elif model == "clip":
            distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

            self.module = EpicKitchensFramesModule(
                csv_file='./data/annotations/processed/train.csv',
                frames_dir = './data/sampled/train',
                tokenizer = distilbert_tokenizer,
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
            )
        self.module.setup('fit')
        
    def train(self):
        self.trainer.fit(self.model, datamodule=self.module)


if __name__ == "__main__":
    trainer = ModelTrainer("experiments/configs/config.yaml", model="clip")
    trainer.train()
    # TODO: CLI 