"""
TRAINING (data batching): wrap questions, targets, and supervision into a PyTorch
dataset and merge multiple questions into one batch

- Qs have different lengths and numbers of candidates - collate function helps to merge them into a batch for PyTorch's DataLoader
    - pads question token IDs into a rectangular tensor
    - flattens pull-node and answer candidates into 1-D tensors
    - 1-D tensors have a parallel "batch_indices" tensor recording which question each candidate belongs to
    - leaves each question's supervision/targets as Python lists (i.e. each question has its own graph)
"""

import torch
from torch.utils.data import Dataset


class QATrainingDataset(Dataset):
    """
    pairs each QA example with its generated training targets and BFS supervision

    - supervision is used to construct the question-specific graph during training
    """


    def __init__(self, examples, targets, supervision):

        if not (len(examples) == len(targets) == len(supervision)):
            raise ValueError(
                "Examples, targets and supervision must have equal lengths."
            )

        self.examples = examples
        self.targets = targets
        self.supervision = supervision


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):

        return {
            "question": self.examples[index]["question"],

            "topic_entity_id": self.targets[index]["topic_entity_id"],
            "relation_targets": self.targets[index]["relation_targets"],

            "pull_candidates": self.targets[index]["pull_candidates"],
            "answer_candidates": self.targets[index]["answer_candidates"],

            "supervision": self.supervision[index],
            "training_targets": self.targets[index],
        }


def collate_qa_batch(batch, vocab):
    """
    combines several QA examples into one batch

    - Qs are padded to the same length
    - pull-node and answer candidates are flattened
        - batch indices used to track which Q each candidate belongs to
    """

    # 1. Encode questions.
    questions = [item["question"] for item in batch]
    tokens, lengths = vocab.encode_batch(questions)

    # 2. Collect topic entity IDs.
    topic_entity_ids = torch.tensor(
        [item["topic_entity_id"] for item in batch],
        dtype=torch.long
    )

    # 3. Stack relation targets.
    relation_targets = torch.stack([
        item["relation_targets"] for item in batch
    ])

    # 4. Flatten pull-node candidates.
    pull_entity_ids = []
    pull_iterations = []
    pull_labels = []
    pull_batch_indices = []

    for batch_index, item in enumerate(batch):
        for candidate in item["pull_candidates"]:
            pull_entity_ids.append(candidate["entity_id"])
            pull_iterations.append(candidate["iteration"])
            pull_labels.append(candidate["label"])
            pull_batch_indices.append(batch_index)

    # 5. Flatten answer candidates.
    answer_entity_ids = []
    answer_labels = []
    answer_batch_indices = []

    for batch_index, item in enumerate(batch):
        for candidate in item["answer_candidates"]:
            answer_entity_ids.append(candidate["entity_id"])
            answer_labels.append(candidate["label"])
            answer_batch_indices.append(batch_index)

    # 6. Return the batch.
    return {
        "question_tokens": tokens,
        "question_lengths": lengths,

        "topic_entity_ids": topic_entity_ids,
        "relation_targets": relation_targets,

        "pull_entity_ids": torch.tensor(
            pull_entity_ids, dtype=torch.long
        ),
        "pull_iterations": torch.tensor(
            pull_iterations, dtype=torch.long
        ),
        "pull_labels": torch.tensor(
            pull_labels, dtype=torch.float32
        ),
        "pull_batch_indices": torch.tensor(
            pull_batch_indices, dtype=torch.long
        ),

        "answer_entity_ids": torch.tensor(
            answer_entity_ids, dtype=torch.long
        ),
        "answer_labels": torch.tensor(
            answer_labels, dtype=torch.float32
        ),
        "answer_batch_indices": torch.tensor(
            answer_batch_indices, dtype=torch.long
        ),

        # Keep these as lists because each question has a
        # different graph and potentially different candidates.
        "supervision": [
            item["supervision"] for item in batch
        ],
        "training_targets": [
            item["training_targets"] for item in batch
        ],
    }