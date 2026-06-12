import argparse
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.trainer import ModelTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--test', action='store_true', help='inference on test sets')
    parser.add_argument('--validate', action='store_true', help='inference on validation set')
    parser.add_argument('--ckpt', action=str, default=None, help='restore weights from checkpoint')

    user_args = parser.parse_args()

    trainer = ModelTrainer(user_args.config)
    
    if user_args.validate:
        trainer.validate(user_args.ckpt)

    if user_args.test:
        trainer.test(user_args.ckpt)