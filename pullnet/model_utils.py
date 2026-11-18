"""
helper file shaused red by train.py and both evaluation scripts
- builds, saves and loads the five model components
    - settings saved inside checkpoint (model_config) and used to rebuild model on load
"""

import torch

from model import QuestionEncoder, RelationScorer, PullNodeScorer, AnswerScorer
from message_passing import QuestionGraphEncoder

COMPONENT_NAMES = [
    "question_encoder",
    "relation_scorer",
    "graph_encoder",
    "pull_scorer",
    "answer_scorer",
]


def build_components(
    kg,
    vocab_size,
    embedding_dim=128,
    word_embedding_dim=100,
    num_rounds=2,
    use_topic_flag=True,
):
    """Create the five (untrained) model components as a dictionary."""
    return {
        "question_encoder": QuestionEncoder(
            vocab_size=vocab_size,
            embedding_dim=word_embedding_dim,
            hidden_dim=embedding_dim,
        ),
        "relation_scorer": RelationScorer(
            num_relations=kg.num_relations, embedding_dim=embedding_dim
        ),
        "graph_encoder": QuestionGraphEncoder(
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
            embedding_dim=embedding_dim,
            num_rounds=num_rounds,
            use_topic_flag=use_topic_flag,
        ),
        "pull_scorer": PullNodeScorer(embedding_dim=embedding_dim),
        "answer_scorer": AnswerScorer(embedding_dim=embedding_dim),
    }


def make_model_config(embedding_dim, word_embedding_dim, num_rounds, use_topic_flag):
    return {
        "embedding_dim": embedding_dim,
        "word_embedding_dim": word_embedding_dim,
        "num_rounds": num_rounds,
        "use_topic_flag": use_topic_flag,
    }


def all_parameters(components):
    """Flat list of every trainable parameter (for the optimiser)."""
    params = []
    for name in COMPONENT_NAMES:
        params.extend(components[name].parameters())
    return params


def set_mode(components, training):
    """switch every component between training and evaluation mode"""
    for name in COMPONENT_NAMES:
        components[name].train(training)


def save_checkpoint(path, components, vocab, epoch, dev_loss, model_config,
                    optimizer=None):
    checkpoint = {
        "epoch": epoch,
        "dev_loss": dev_loss,
        "vocab": vocab,
        "model_config": model_config,
    }
    for name in COMPONENT_NAMES:
        checkpoint[name] = components[name].state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, kg, device="cpu"):
    """
    rebuild model from a checkpoint and put it in evaluation mode

    - returns (components, vocab, checkpoint)
    weights_only=False is needed because the vocabulary object is pickled
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    config = checkpoint.get("model_config")
    if config is None:
        raise RuntimeError(
            f"{path} has no 'model_config' (saved by an older train.py). "
            "The architecture changed, so please retrain."
        )

    vocab = checkpoint["vocab"]
    components = build_components(
        kg,
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        word_embedding_dim=config["word_embedding_dim"],
        num_rounds=config["num_rounds"],
        use_topic_flag=config["use_topic_flag"],
    )

    for name in COMPONENT_NAMES:
        components[name].load_state_dict(checkpoint[name])
        components[name].to(device)

    set_mode(components, training=False)
    return components, vocab, checkpoint