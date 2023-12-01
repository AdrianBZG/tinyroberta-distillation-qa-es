import argparse
import sys
import logging
import json
import os
import torch
from transformers import AutoConfig, AutoModel

logging.basicConfig(level=logging.INFO,
                    format='[create_tiny_model:%(levelname)s] %(message)s')


def get_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',
                        default="stevemobs/roberta-large-fine-tuned-squad-es",
                        required=True,
                        help='Base model from which to create its tiny version')

    parser.add_argument('--config_path',
                        default="configs/tinyroberta.json",
                        required=True,
                        help='Path to the json config with the parameters to change on the base model')

    parser.add_argument('--output_dir',
                        default="models/tinyroberta-squad-es",
                        required=True,
                        help='Path to the json config with the parameters to change on the base model')

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config_path):
        raise ValueError(f'Path to config {args.config_path} does not exist.')

    logging.info(f"Getting config for base_model: {args.base_model}")
    config = AutoConfig.from_pretrained(args.base_model)
    logging.info(f"Base model config: {config}")

    logging.info(f"Loading parameters to change on base model from: {args.config_path}")
    with open(args.config_path) as file:
        try:
            params_to_change = json.load(file)
        except Exception as e:
            logging.error(e)

    logging.info(f"Parameters to set in base model: {params_to_change}")
    config.update(params_to_change)
    logging.info(f"Final config to load the model with: {config}")

    model = AutoModel.from_pretrained(args.base_model, config=config, torch_dtype=torch.float16)
    logging.info(f"Loaded model has {get_model_size(model)} parameters")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, from_pt=True)
