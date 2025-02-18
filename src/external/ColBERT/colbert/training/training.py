import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from datasets import load_dataset
from sentence_transformers import InputExample

# grab locally
from external.ColBERT.colbert.training.utils import print_progress, manage_checkpoints
from finetune.colbert.colbert_pairwise_evaluator import PairwiseEvaluator

def convert_dataset_to_triplets(df, evaluation: bool = False):
    all_instances = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if not evaluation:
            all_instances.extend([
                InputExample(texts=[row['q1'], row['doc1']], label=1),
                InputExample(texts=[row['q1'], row['doc2']], label=0),
                InputExample(texts=[row['q2'], row['doc2']], label=1),
                InputExample(texts=[row['q2'], row['doc1']], label=0),
            ])
        else:
            all_instances.extend([
                InputExample(texts=[row["q1"], row["q2"], row["doc1"], row["doc2"]]),
            ])
    print("Number of instances:", len(all_instances))
    return all_instances

def train(args):
    print("BHEEWRASFASFAFSF")
    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    print("Loading data...")

    dataset_test = load_dataset("orionweller/NevIR", split="test")
    test_df = dataset_test.to_pandas()

    dataset_validation = load_dataset("orionweller/NevIR", split="validation")
    validation_df = dataset_validation.to_pandas()

    print("Loading evaluation...")
    validation_data = convert_dataset_to_triplets(validation_df, evaluation=True) # NevIR to triplets validation
    evaluator = PairwiseEvaluator.from_input_examples(validation_data, name="validation", model_name='colbert') # triplet evaluator

    test_data = convert_dataset_to_triplets(test_df, evaluation=True)
    evaluator_test = PairwiseEvaluator.from_input_examples(test_data, name="test", model_name='colbert')

    # XXX sien added this outer epoch loop
    num_epochs = 20

    # Training loop
    for epoch in range(num_epochs):  # Outer loop for epochs
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
        train_loss = 0.0
        colbert.train()
        
        # If the reader needs resetting for each epoch, uncomment the following line:
        # reader = initialize_reader(...)  # Reset or reinitialize the reader if required

        if args.lazy:
            reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
        else:
            reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

        for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
            this_batch_loss = 0.0


            for queries, passages in BatchSteps:
                with amp.context():
                    scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                    loss = criterion(scores, labels[:scores.size(0)])
                    loss = loss / args.accumsteps

                if args.rank < 1:
                    print_progress(scores)

                amp.backward(loss)

                train_loss += loss.item()
                this_batch_loss += loss.item()

            amp.step(colbert, optimizer)

            if args.rank < 1:
                avg_loss = train_loss / (batch_idx+1)

                num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
                elapsed = float(time.time() - start_time)

                log_to_mlflow = (batch_idx % 20 == 0)
                Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
                Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

                print_message(batch_idx, avg_loss)
                # sien edited this: manage_checkpoints(args, colbert, optimizer, batch_idx+1)
                manage_checkpoints(args, colbert, optimizer, batch_idx+1, epoch)
            