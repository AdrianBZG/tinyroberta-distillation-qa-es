import argparse
import sys
import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logging.basicConfig(level=logging.INFO,
                    format='[upload_model:%(levelname)s] %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/evaluation1.json",
                        required=True,
                        help='Path to the json config with the parameters for running evaluation of a model on Squad')

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config_path):
        raise ValueError(f'Path to config {args.config_path} does not exist.')

    logging.info(f"Loading parameters for upload from: {args.config_path}")
    with open(args.config_path) as file:
        try:
            config_args = json.load(file)
        except Exception as e:
            logging.error(e)

    # Upload model
    REPO_NAME = ""
    MODEL_PATH = 'models/tinyroberta-squad-es-distilled'

    torch_dtype = torch.float16 if config_args['fp16'] else torch.float32

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH,
                                                          torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.push_to_hub(REPO_NAME)
    tokenizer.push_to_hub(REPO_NAME)
