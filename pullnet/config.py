"""
Shared settings used by training and both evaluation scripts

Central file to ensure train.py and evaluate.py both access the same file name
"""

from pathlib import Path

# ---- Data -------------------------------------------------------------
HOP = 2
DATA_DIR = "."               # folder contains 1-hop, 2-hop, 3-hop
KB_PATH = "kb.txt"

# ---- Model ------------------------------------------------------------
WORD_EMBEDDING_DIM = 100     # word vector size in the q LSTM
EMBEDDING_DIM = 128          # size of question / entity / relation vectors
NUM_MESSAGE_ROUNDS = 2       # 2 = information can travel 2 hops
USE_TOPIC_FLAG = True        # mark topic entity inside graph encoder

# ---- Retrieval (used at inference time) --------------------------------
MAX_FACTS_PER_ENTITY = 10    # number of facts retrieved every time an entity is expanded
TOP_K_PULL_NODES = 10        # number of entities chosen for expansion after iteration 1

# ---- Training-graph construction --------------------------------------
# Upper bound on "filler" edges per expanded entity in the training graph
# aligns with MAX_FACTS_PER_ENTITY at inference time.
TRAIN_MAX_EDGES_PER_PARENT = 20

# ---- Files ------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints")

def checkpoint_path(hop=HOP):
    """Source of truth for where the best model is saved/loaded."""
    return CHECKPOINT_DIR / f"hop_{hop}_best.pt"

# QUESTIONS
# is this file called by otehr files? how does setting these configs work?
# what does it mean for info to only travel 2 hops? how does that work?
# max facts per entity = number of connections to explore via an edge?
# i dont understand train max edges per parent