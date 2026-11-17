# TO UPDATE COMMENTS

"""
TRAINING: train all five components together and save the best checkpoint.

Quick start
-----------
  python train.py --train-examples 500 --dev-examples 200 --epochs 1   # smoke test
  python train.py --train-examples 20000 --dev-examples 2000           # real run

How a run is organised
----------------------
  1. Load the KB and the 2-hop QA data; build the question vocabulary.
  2. Shuffle the training questions once (fixed seed) and optionally keep a subset.
  3. Each epoch walks through the data in CHUNKS of questions. For each chunk:
        BFS supervision -> training targets -> DataLoader -> optimiser steps.
     Chunking exists so we never hold BFS results for the whole dataset in memory.
  4. After every epoch evaluate the loss on the dev set; save the checkpoint when
     it improves; stop early if it fails to improve `--patience` times.

Dev loss is only used to choose the checkpoint. The real quality measure is
evaluate_inference.py (Hits@K on the test split).
"""

import argparse
import gc
import random
import time
from collections import OrderedDict
from functools import partial

import torch
from torch.utils.data import DataLoader

import config
from data import load_knowledge_graph, load_all_datasets
from graph import KnowledgeGraph
from model import QuestionVocabulary
from supervision import build_supervision_dataset
from training_data import build_training_targets_dataset
from dataset import QATrainingDataset, collate_qa_batch
from training import compute_batch_losses
from model_utils import (
    build_components, make_model_config, all_parameters, set_mode,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hop", type=int, default=config.HOP)
parser.add_argument("--train-examples", type=int, default=20000,
                    help="Number of training questions (0 = the whole split).")
parser.add_argument("--dev-examples", type=int, default=2000,
                    help="Number of dev questions (0 = the whole split).")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--patience", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--chunk-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--rounds", type=int, default=config.NUM_MESSAGE_ROUNDS)
parser.add_argument("--no-topic-flag", action="store_true")
parser.add_argument("--max-edges-per-parent", type=int,
                    default=config.TRAIN_MAX_EDGES_PER_PARENT)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

HOP = args.hop
DEVICE = torch.device(args.device)
use_topic_flag = not args.no_topic_flag

torch.manual_seed(args.seed)
random.seed(args.seed)

config.CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = config.checkpoint_path(HOP)

print("Device:", DEVICE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading knowledge graph...")
kg = KnowledgeGraph(load_knowledge_graph(config.KB_PATH))
print(f"  {kg.num_entities} entities, {kg.num_facts} facts, "
      f"{kg.num_relations} direction-aware relations")

datasets = load_all_datasets(config.DATA_DIR, hops=(HOP,))
all_train = list(datasets[HOP]["train"])
all_dev = list(datasets[HOP]["dev"])


def subsample(examples, n, seed):
    """Shuffle once with a fixed seed, then keep the first n (0 = keep all).
    Shuffling first matters: the files may be ordered by question template, so
    'the first n lines' would be a biased subset."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    return examples if n == 0 else examples[:n]


train_examples = subsample(all_train, args.train_examples, args.seed)
dev_examples = subsample(all_dev, args.dev_examples, args.seed + 1)
print(f"  train: {len(train_examples)}/{len(all_train)}  "
      f"dev: {len(dev_examples)}/{len(all_dev)}")

# ---------------------------------------------------------------------------
# 2. Vocabulary (from the full training split only - never dev/test)
# ---------------------------------------------------------------------------
vocab = QuestionVocabulary()
vocab.build([e["question"] for e in all_train])
print("Vocabulary size:", len(vocab))

# ---------------------------------------------------------------------------
# 3. Model + optimiser
# ---------------------------------------------------------------------------
components = build_components(
    kg,
    vocab_size=len(vocab),
    embedding_dim=config.EMBEDDING_DIM,
    word_embedding_dim=config.WORD_EMBEDDING_DIM,
    num_rounds=args.rounds,
    use_topic_flag=use_topic_flag,
)
for module in components.values():
    module.to(DEVICE)

model_config = make_model_config(
    config.EMBEDDING_DIM, config.WORD_EMBEDDING_DIM, args.rounds, use_topic_flag
)
parameters = all_parameters(components)
optimizer = torch.optim.Adam(parameters, lr=args.lr)
print("Trainable parameters:", sum(p.numel() for p in parameters))

collate_fn = partial(collate_qa_batch, vocab=vocab)

# BFS results are reused across chunks and epochs (many questions share a topic).
bfs_cache = OrderedDict()
BFS_CACHE_SIZE = 256


# ---------------------------------------------------------------------------
# 4. One pass over a DataLoader
# ---------------------------------------------------------------------------
def run_epoch(loader, training):
    """Average losses over a DataLoader; update weights only if training."""
    set_mode(components, training)
    sums = {"total": 0.0, "relation": 0.0, "pull": 0.0, "answer": 0.0}
    count = 0

    for batch in loader:
        n = len(batch["question_lengths"])

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            total, relation, pull, answer = compute_batch_losses(
                batch=batch, kg=kg, device=DEVICE,
                max_edges_per_parent=args.max_edges_per_parent,
                **components,
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()

        sums["total"] += total.item() * n
        sums["relation"] += relation.item() * n
        sums["pull"] += pull.item() * n
        sums["answer"] += answer.item() * n
        count += n

    return {k: v / count for k, v in sums.items()}


# ---------------------------------------------------------------------------
# 5. One epoch, processed chunk by chunk
# ---------------------------------------------------------------------------
def run_chunked_epoch(examples, training, epoch_number):
    chunks = [
        (i, examples[start:start + args.chunk_size])
        for i, start in enumerate(range(0, len(examples), args.chunk_size))
    ]
    if training:  # different chunk order every epoch
        random.Random(args.seed + epoch_number).shuffle(chunks)

    name = "Train" if training else "Dev"
    sums = {"total": 0.0, "relation": 0.0, "pull": 0.0, "answer": 0.0}
    seen = 0
    reachable = gold = 0
    start_time = time.time()

    print(f"\n{name}: {len(examples)} questions in {len(chunks)} chunks")

    for position, (chunk_index, chunk) in enumerate(chunks, start=1):

        # BFS supervision -> sampled targets (seeded by chunk index so the
        # sampled negatives are the same every epoch).
        supervision, stats = build_supervision_dataset(
            kg, chunk, max_hops=HOP,
            bfs_cache_size=BFS_CACHE_SIZE, bfs_cache=bfs_cache,
        )
        targets = build_training_targets_dataset(
            supervision, max_hops=HOP, num_relations=kg.num_relations,
            random_seed=args.seed + chunk_index,
        )

        loader = DataLoader(
            QATrainingDataset(chunk, targets, supervision),
            batch_size=args.batch_size, shuffle=training, collate_fn=collate_fn,
        )
        losses = run_epoch(loader, training)

        for key in sums:
            sums[key] += losses[key] * len(chunk)
        seen += len(chunk)
        gold += stats["total_gold_answers"]
        reachable += stats["total_reachable_answers"]

        # Free this chunk's memory before preparing the next.
        del loader, targets, supervision
        gc.collect()

        # Progress with timing, so a slow run is visible straight away.
        elapsed = time.time() - start_time
        eta = elapsed / position * (len(chunks) - position)
        print(f"  {name} chunk {position}/{len(chunks)} | "
              f"loss {sums['total'] / seen:.4f} | "
              f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
              flush=True)

    return {k: v / seen for k, v in sums.items()}, reachable, gold


# ---------------------------------------------------------------------------
# 6. Training loop with early stopping
# ---------------------------------------------------------------------------
best_dev_loss = float("inf")
best_epoch = 0
bad_epochs = 0

for epoch in range(args.epochs):
    print("\n" + "=" * 60 + f"\nEPOCH {epoch + 1}/{args.epochs}\n" + "=" * 60)

    train_losses, tr_reach, tr_gold = run_chunked_epoch(train_examples, True, epoch)
    with torch.no_grad():
        dev_losses, dv_reach, dv_gold = run_chunked_epoch(dev_examples, False, epoch)

    for label, l in (("Train", train_losses), ("Dev  ", dev_losses)):
        print(f"  {label} loss: {l['total']:.4f} (relation {l['relation']:.4f}, "
              f"pull {l['pull']:.4f}, answer {l['answer']:.4f})")
    print(f"  Gold answers reachable within {HOP} hops: "
          f"train {tr_reach}/{tr_gold}, dev {dv_reach}/{dv_gold}")

    if dev_losses["total"] < best_dev_loss:
        best_dev_loss, best_epoch, bad_epochs = dev_losses["total"], epoch + 1, 0
        save_checkpoint(CHECKPOINT_PATH, components, vocab, epoch + 1,
                        best_dev_loss, model_config, optimizer)
        print(f"  Saved best checkpoint -> {CHECKPOINT_PATH}")
    else:
        bad_epochs += 1
        print(f"  Dev loss did not improve ({bad_epochs}/{args.patience})")
        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

print("\n" + "=" * 60)
print(f"TRAINING COMPLETE | best epoch {best_epoch} | "
      f"best dev loss {best_dev_loss:.4f} | {CHECKPOINT_PATH}")
