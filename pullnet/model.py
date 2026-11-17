
# TO UPDATE COMMENTS

"""
MODEL (neural components other than the graph encoder).

Five pieces, in the order data flows through them:

  QuestionVocabulary  words  -> integer IDs
  QuestionEncoder     IDs    -> ONE 128-d vector summarising the question (LSTM)
  RelationScorer      question vector -> a 0..1 score for each of the 18 relations
                      ("which kinds of facts are worth retrieving?")
  PullNodeScorer      (entity vector, question vector) -> "expand this entity next?"
  AnswerScorer        (entity vector, question vector) -> "is this entity an answer?"

The three scorers are separate because they answer different questions: an
actor can be a great entity to expand (pull) without being the answer.
"""
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class QuestionVocabulary:
    """
    Converts question words into integer IDs.

    Each unique token receives one ID. The IDs are then used to look
    up trainable token embeddings in the QuestionEncoder.
    """

    def __init__(self):
        # Reserve two IDs for special tokens:
        # 0 = padding, used to make questions in a batch the same length.
        # 1 = unknown, used for words not seen when building the vocabulary.
        self.token_to_id = {
            "<pad>": 0,
            "<unk>": 1
        }

        self.pad_id = 0
        self.unk_id = 1

    def tokenize(self, question):
        """
        Split a question into tokens.

        Lowercase the text so that, for example, "Film" and "film"
        are treated as the same token.

        The [topic entity] is replaced by a single placeholder token <ent>.
        Why: MetaQA has ~43k entity names, so keeping them as words would
        (a) make the vocabulary huge, (b) give the LSTM thousands of one-off
        words it can never learn from, and (c) hide the real signal, which is
        the QUESTION PATTERN ("who directed the films starring <ent>"). The
        topic entity's identity is supplied by the graph, not by the question text.
        """
        question = re.sub(r"\[[^\]]*\]", " <ent> ", question)
        return re.findall(r"<ent>|[a-z0-9]+", question.lower())

    def build(self, questions):
        """
        Build the vocabulary from training questions.

        Each new token receives the next available integer ID.
        We only build the vocabulary from training questions so that
        validation and test data do not influence vocabulary construction.
        """
        for question in questions:
            tokens = self.tokenize(question)

            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)

    def encode(self, question):
        """
        Convert one question into a list of token IDs.

        A token not in the vocabulary is mapped to <unk>.
        """
        tokens = self.tokenize(question)

        return [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

    def encode_batch(self, questions):
        """
        Convert multiple questions into a padded tensor.

        Questions have different numbers of tokens. Padding makes
        them the same length so they can be processed together.

        Returns:
            token_ids: Tensor of shape (batch_size, max_sequence_length)
            lengths:   Tensor containing each question's actual length
        """
        encoded_questions = [
            self.encode(question)
            for question in questions
        ]

        # Record the original lengths before padding.
        lengths = torch.tensor(
            [len(ids) for ids in encoded_questions],
            dtype=torch.long
        )

        # Find the longest question in this batch.
        max_length = max(lengths).item()

        # Fill the batch with padding IDs.
        token_ids = torch.full(
            (len(questions), max_length),
            fill_value=self.pad_id,
            dtype=torch.long
        )

        # Copy each question's actual token IDs into its row.
        for i, ids in enumerate(encoded_questions):
            token_ids[i, :len(ids)] = torch.tensor(ids)

        return token_ids, lengths

    def __len__(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)


class QuestionEncoder(nn.Module):
    """
    Converts a batch of questions into fixed-size vectors using:

        token IDs
            -> token embeddings
            -> LSTM
            -> final hidden state

    The final output has shape (batch_size, hidden_dim).
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        pad_id=0
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Look up a trainable numerical vector for each token ID.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        # Process the token embeddings in their original sequence order.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, token_ids, lengths):
        """
        Args:
            token_ids: Integer tensor with shape
                       (batch_size, sequence_length)

            lengths:   Actual number of tokens in each question.
                       Padding tokens are not included.

        Returns:
            question_vectors: Tensor with shape
                              (batch_size, hidden_dim)
        """

        # Step 1: Replace every token ID with its embedding vector.
        # Shape:
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(token_ids)

        # Step 2: Pack the sequences so the LSTM does not waste
        # computation processing padding tokens.
        #
        # enforce_sorted=False means we do not need to sort questions
        # by length before passing them into the LSTM.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Step 3: Process the packed sequences using the LSTM.
        #
        # hidden has shape:
        # (num_layers, batch_size, hidden_dim)
        #
        # cell has the same shape, but represents the LSTM cell state.
        _, (hidden, cell) = self.lstm(packed)

        # Step 4: Extract the final hidden state from the last LSTM layer.
        #
        # Shape changes from:
        # (num_layers, batch_size, hidden_dim)
        # to:
        # (batch_size, hidden_dim)
        #
        # This is our fixed-size representation of each question.
        question_vectors = hidden[-1]

        return question_vectors












class RelationScorer(nn.Module):
    """
    Scores each possible relation against a question representation.

    The question vector and relation embeddings must have the same
    dimension so that we can compare them using a dot product.
    """

    def __init__(
        self,
        num_relations,
        embedding_dim=128
    ):
        super().__init__()

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Give each relation a trainable embedding vector.
        #
        # Shape:
        # (num_relations, embedding_dim)
        self.relation_embeddings = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim
        )

    def forward(self, question_vectors):
        """
        Args:
            question_vectors:
                Tensor of shape (batch_size, embedding_dim)

        Returns:
            relation_logits:
                Tensor of shape (batch_size, num_relations)

            relation_scores:
                Sigmoid-transformed scores in the range (0, 1).
        """

        # Compare every question vector with every relation embedding.
        #
        # question_vectors:
        # (batch_size, embedding_dim)
        #
        # relation_embeddings.weight:
        # (num_relations, embedding_dim)
        #
        # Transposing the relation matrix gives:
        # (embedding_dim, num_relations)
        #
        # The matrix multiplication therefore produces:
        # (batch_size, num_relations)
        relation_logits = torch.matmul(
            question_vectors,
            self.relation_embeddings.weight.T
        )

        # Convert raw scores into values between 0 and 1.
        relation_scores = torch.sigmoid(relation_logits)

        return relation_logits, relation_scores












class PullNodeScorer(nn.Module):
    """
    Estimates whether an entity should be expanded for a question.

    It combines the current entity representation with the question
    representation and produces one pull score for each entity.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.embedding_dim = embedding_dim

        # The entity and question vectors are concatenated.
        # Each has embedding_dim values, so the input has 2 * embedding_dim.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, entity_vectors, question_vectors):
        """
        Args:
            entity_vectors:
                Shape (batch_size, embedding_dim)

            question_vectors:
                Shape (batch_size, embedding_dim)

        Returns:
            pull_logits:
                Raw scores, shape (batch_size,)

            pull_scores:
                Sigmoid scores between 0 and 1,
                shape (batch_size,)
        """

        # Combine the entity's representation and the question vector.
        combined = torch.cat(
            [entity_vectors, question_vectors],
            dim=1
        )

        # Predict one raw pull score per entity.
        pull_logits = self.classifier(combined).squeeze(-1)

        # Convert raw scores into values between 0 and 1.
        pull_scores = torch.sigmoid(pull_logits)

        return pull_logits, pull_scores











class AnswerScorer(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.embedding_dim = embedding_dim

        # The scorer considers both:
        # 1. The entity's contextual graph representation.
        # 2. The representation of the question.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, entity_vectors, question_vectors):

        # Combine each entity representation with its
        # corresponding question representation.
        combined = torch.cat(
            [entity_vectors, question_vectors],
            dim=1
        )

        # One raw answer score (logit) per entity.
        answer_logits = self.classifier(combined).squeeze(-1)

        # Convert logits into scores between 0 and 1.
        answer_scores = torch.sigmoid(answer_logits)

        return answer_logits, answer_scores
