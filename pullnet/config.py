"""
Shared settings used by training and both evaluation scripts

Central file to ensure train.py and evaluate.py both access the same file name
"""

from pathlib import Path

# data
HOP = 2
DATA_DIR = "."               # folder contains 1-hop, 2-hop, 3-hop
KB_PATH = "kb.txt"

# model
WORD_EMBEDDING_DIM = 100     # word vector size in the q LSTM
EMBEDDING_DIM = 128          # size of question / entity / relation vectors
NUM_MESSAGE_ROUNDS = 2       # 2 = information can travel 2 hops
USE_TOPIC_FLAG = True        # mark topic entity inside graph encoder

# retrieval (used at inference time)
MAX_FACTS_PER_ENTITY = 10    # number of facts retrieved every time an entity is expanded
TOP_K_PULL_NODES = 10        # number of entities chosen for expansion after iteration 1

# training-graph construction

# upper bound on "filler" edges per expanded entity in the training graph
# aligns with MAX_FACTS_PER_ENTITY at inference time
TRAIN_MAX_EDGES_PER_PARENT = 20

# files
CHECKPOINT_DIR = Path("checkpoints")

def checkpoint_path(hop=HOP):
    """Source of truth for where the best model is saved/loaded."""
    return CHECKPOINT_DIR / f"hop_{hop}_best.pt"