import argparse
import os

from neural_mesh_simplification.trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Neural Mesh Simplification model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    return parser.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config["data"]["data_path"] = args.data_path
    config["training"]["checkpoint_dir"] = args.checkpoint_dir

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    except Exception as e:
        trainer.handle_error(e)
        trainer.save_training_state(os.path.join(args.checkpoint_dir, "training_state.pth"))


if __name__ == "__main__":
    main()
