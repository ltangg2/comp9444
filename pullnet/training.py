"""
TRAINING (loss computation): run one batch through the model and compute 3 losses

the model is trained on three tasks at once (multi-task learning), all with
binary cross-entropy:
    - relation loss = which of the 18 relation types are relevant to this question
        - labels: relations on the BFS shortest paths to the answers
    - pull loss = which entities at distance >= 1 should be expanded next
        - labels: entities on shortest paths = 1, sampled others = 0
    - answer loss = is this candidate entity a correct answer
        - labels: gold answers = 1, sampled non-answers = 0

teacher forcing = the graph each question is encoded on is built from the BFS
supervision (training_utils.build_training_subgraph), not from the model's own
retrieval
"""

import torch
import torch.nn as nn

from training_utils import build_training_subgraph


def _score_candidates(entity_to_row, entity_vectors, question_vector, scorer,
                      candidate_ids, device):
    """Look up the graph vectors of candidate entities and score them."""
    rows = torch.tensor(
        [entity_to_row[int(entity_id)] for entity_id in candidate_ids],
        dtype=torch.long,
        device=device,
    )
    vectors = entity_vectors[rows]
    question_vectors = question_vector.unsqueeze(0).expand(len(vectors), -1)
    logits, _ = scorer(vectors, question_vectors)
    return logits


def compute_batch_losses(
    batch,
    kg,
    graph_encoder,
    question_encoder,
    relation_scorer,
    pull_scorer,
    answer_scorer,
    device="cpu",
    max_edges_per_parent=None,
):
    """
    returns (total_loss, relation_loss, pull_loss, answer_loss)

    - `device` = where the model lives
        - tensors that feed the network are moved to `device`
    """

    # encode all questions in the batch (batch_size, 128)
    question_vectors = question_encoder(
        batch["question_tokens"].to(device),
        batch["question_lengths"],
    )

    # relation prediction: one multi-hot target per question
    relation_logits, _ = relation_scorer(question_vectors)
    relation_loss = nn.BCEWithLogitsLoss()(
        relation_logits, batch["relation_targets"].to(device)
    )

    pull_criterion = nn.BCEWithLogitsLoss()
    answer_criterion = nn.BCEWithLogitsLoss()
    pull_losses, answer_losses = [], []

    subgraph_kwargs = {}
    if max_edges_per_parent is not None:
        subgraph_kwargs["max_edges_per_parent"] = max_edges_per_parent

    # graph differs per question (each Q processed one by one)
    for batch_index, supervision in enumerate(batch["supervision"]):

        question_vector = question_vectors[batch_index]
        targets = batch["training_targets"][batch_index]

        subgraph = build_training_subgraph(kg, supervision, targets, **subgraph_kwargs)

        entity_ids, entity_vectors = graph_encoder(
            kg, subgraph, question_vector,
            topic_entity_id=supervision["topic_entity_id"],
        )
        entity_to_row = {entity_id: row for row, entity_id in enumerate(entity_ids)}

        # pull-node loss
        mask = batch["pull_batch_indices"] == batch_index
        pull_ids = batch["pull_entity_ids"][mask]
        pull_labels = batch["pull_labels"][mask]
        pull_iterations = batch["pull_iterations"][mask]

        # iteration 0 = topic entity (always expanded)
        # no decision to learn - training starts from iteration 1
        keep = pull_iterations > 0
        pull_ids, pull_labels = pull_ids[keep], pull_labels[keep]

        if len(pull_ids) > 0:
            logits = _score_candidates(
                entity_to_row, entity_vectors, question_vector,
                pull_scorer, pull_ids, device,
            )
            pull_losses.append(pull_criterion(logits, pull_labels.to(device)))

        # answer loss
        mask = batch["answer_batch_indices"] == batch_index
        answer_ids = batch["answer_entity_ids"][mask]
        answer_labels = batch["answer_labels"][mask]

        if len(answer_ids) > 0:
            logits = _score_candidates(
                entity_to_row, entity_vectors, question_vector,
                answer_scorer, answer_ids, device,
            )
            answer_losses.append(answer_criterion(logits, answer_labels.to(device)))

    # average over questions - a batch may have no pull decisions at all
    # (e.g. 1-hop) use a zero still attached to the graph
    zero = question_vectors.sum() * 0.0
    pull_loss = torch.stack(pull_losses).mean() if pull_losses else zero
    answer_loss = torch.stack(answer_losses).mean() if answer_losses else zero

    # equal weights for the three tasks
    total_loss = relation_loss + pull_loss + answer_loss
    return total_loss, relation_loss, pull_loss, answer_loss