# Codes adapted from https://github.com/rimon15/evil_twins

from training import Imprompter
import utils as utils
import torch

import os
from typing import Optional
from argparse import ArgumentParser
from pathlib import Path
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--exp_name", type=str, default="exp_undefined")
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--n_proposals", type=int, default=1)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--syntax_weight", type=float, default=0)
    parser.add_argument("--ppl_preference", type=float, default=1)
    parser.add_argument("--mask_unrecognizable_unicode", type=bool, default=True)
    parser.add_argument("--vocab", type=str, default="non_english", choices = ["english", "non_english", "all_allow", "hybrid"])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--start_from_file", type=str, default='')
    parser.add_argument("--initial_suffix", type=str, default='!!!!!!!!!!!!!', required=True)
    parser.add_argument("--reuse_log", action="store_true", help="dependent on start_from_file")
    parser.add_argument("--autorestart", type=int, default=50)

    args = parser.parse_args()

    assert args.subset_size % args.batch_size == 0
    
    model, tokenizer = utils.load_model_tokenizer(
            args.model_name, True, args.sharded
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = json.load(open(args.dataset_path, "r"))

    from datetime import datetime
    now = datetime.now()
    outfile_prefix = os.path.join(args.output_dir, f"{args.exp_name}_{now.strftime('%b')}_{now.day}_{now.hour}_{now.minute}".replace(' ', '_'))
    with open(outfile_prefix+'.log', 'w') as f:
        f.write(f"{args}\n")

    trainer_args = {
        "reuse_log": args.reuse_log,
        "autorestart": args.autorestart,
        "syntax_weight": args.syntax_weight,
        "mask_unrecognizable_unicode": args.mask_unrecognizable_unicode,
        "ppl_preference": args.ppl_preference,
    }
    trainer = Imprompter(
        model,
        tokenizer,
        args.batch_size,
        args.num_epochs,
        args.top_k,
        args.n_proposals,
        args.subset_size,
        args.initial_suffix,
        args.vocab,
        outfile_prefix,
        args.start_from_file,
        **trainer_args,
    )
    trainer.load_dataset(dataset)

    print("Running Optimization")
    print(f"Number of prompts: {len(dataset)}")
    print(f"Number of trials: {args.n_trials}")

    if args.sharded:
       del model
       utils.free_cuda_memory()

    for i in range(args.n_trials):
        trainer.train(trial = i)


