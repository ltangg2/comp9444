"""
MODEL (neural components other than the graph encoder)

  - QuestionVocabulary: converts words to integer IDs
  - QuestionEncoder: converts IDs to one 128-d vector summarising the Q (LSTM)
  - RelationScorer: question vector = a 0..1 score for each of the 18 relations
    - checks which kinds of facts are worth retrieving
  - PullNodeScorer: (entity vector, question vector)
    - check whether to expand this entity next
  - AnswerScorer: (entity vector, question vector)
    - check if this entity is an answer

separate scorers as they answer different components of the Q (E.g. an
actor can be an entity to expand without being the answer)
"""
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class QuestionVocabulary:
    """
    converts question words into integer IDs

    - 1 ID per unique token
    - IDs are used to look up trainable token embeddings in the QuestionEncoder
    """


    def __init__(self):
        # reserve two IDs for special tokens:
        # 0 = padding (makes questions in a batch the same length)
        # 1 = unknown (used for words not seen when building the vocabulary)
        self.token_to_id = {
            "<pad>": 0,
            "<unk>": 1
        }

        self.pad_id = 0
        self.unk_id = 1


    def tokenize(self, question):
        """
        split a question into tokens

        - topic entity is replaced by a single placeholder token <ent>
            - ~43k entity names in MetaQA, too large to keep as words
        """
        question = re.sub(r"\[[^\]]*\]", " <ent> ", question)
        return re.findall(r"<ent>|[a-z0-9]+", question.lower())


    def build(self, questions):
        """
        build the vocabulary from training questions

        - each new token is assigned the next avaialble integer ID
        """
        for question in questions:
            tokens = self.tokenize(question)

            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)


    def encode(self, question):
        """
        convert one question into a list of token IDs

        - a token not in the vocabulary is mapped to <unk>
        """
        tokens = self.tokenize(question)

        return [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]


    def encode_batch(self, questions):
        """
        convert multiple questions into a padded tensor

        - padding gives all Qs the same length to be processed together
            - Qs would otherwise all have different numbers of tokens
        - returns:
            - token_ids: tensor of shape (batch_size, max_sequence_length)
            lengths: tensor containing each Q's actual length
        """
        encoded_questions = [
            self.encode(question)
            for question in questions
        ]

        # record original lengths before padding
        lengths = torch.tensor(
            [len(ids) for ids in encoded_questions],
            dtype=torch.long
        )

        max_length = max(lengths).item()

        # fill batch with padding IDs
        token_ids = torch.full(
            (len(questions), max_length),
            fill_value=self.pad_id,
            dtype=torch.long
        )

        # copy each Q's actual token IDs into its row
        for i, ids in enumerate(encoded_questions):
            token_ids[i, :len(ids)] = torch.tensor(ids)

        return token_ids, lengths


    def __len__(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)


class QuestionEncoder(nn.Module):
    """
    converts a batch of questions into fixed-size vectors using token IDs (token embeddings, LSTM, inal hidden state)

    final output has shape (batch_size, hidden_dim)
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

        # look up a trainable numerical vector for each token ID
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        # process the token embeddings in their original sequence order
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )


    def forward(self, token_ids, lengths):
        """
        parameters:
            - token_ids: integer tensor with shape (batch_size, sequence_length)
            lengths: actual number of tokens in each question (excluding padding)

        returns:
            - question_vectors: tensor with shape (batch_size, hidden_dim)
        """

        # replace every token ID with its embedding vector
        # shape: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(token_ids)

        # pack the sequences so LSTM does not waste computation processing padding tokens
        # enforce_sorted=False = do not need to sort Qs by length before passing into LSTM
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # process the packed sequences using the LSTM
        # hidden has shape: (num_layers, batch_size, hidden_dim)
        # cell has the same shape and represents the LSTM cell state
        _, (hidden, cell) = self.lstm(packed)

        # extract final hidden state from the last LSTM layer
        # shape becomes: (batch_size, hidden_dim) = fixed-size representation of each question.
        question_vectors = hidden[-1]

        return question_vectors


class RelationScorer(nn.Module):
    """
    scores each possible relation against a question representation

    - Q vector and relation embeddings must have the same dimension
        - can compare them using a dot product
    """


    def __init__(
        self,
        num_relations,
        embedding_dim=128
    ):
        super().__init__()

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # give each relation a trainable embedding vector
        # shape: (num_relations, embedding_dim)
        self.relation_embeddings = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim
        )


    def forward(self, question_vectors):
        """
        parameters:
            - question_vectors: tensor of shape (batch_size, embedding_dim)

        returns:
            - relation_logits: tensor of shape (batch_size, num_relations)
            - relation_scores: sigmoid-transformed scores in the range (0, 1)
        """

        # compare every question vector with every relation embedding
        #
        # question_vectors: batch_size, embedding_dim)
        # relation_embeddings.weight: (num_relations, embedding_dim)
        # transposing the relation matrix gives: (embedding_dim, num_relations)
        # matrix multiplication produces: (batch_size, num_relations)
        relation_logits = torch.matmul(
            question_vectors,
            self.relation_embeddings.weight.T
        )

        # convert raw scores into values between 0 and 1
        relation_scores = torch.sigmoid(relation_logits)

        return relation_logits, relation_scores


class PullNodeScorer(nn.Module):
    """
    estimates whether an entity should be expanded for a question

    - combines the current entity representation with the question
    representation and produces one pull score for each entity
    """


    def __init__(self, embedding_dim=128):
        super().__init__()

        self.embedding_dim = embedding_dim

        # entity and question vectors are concatenated
        # each has embedding_dim values - the input has 2 * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )


    def forward(self, entity_vectors, question_vectors):
        """
        parameters:
            - entity_vectors: shape (batch_size, embedding_dim)
            - question_vectors: shape (batch_size, embedding_dim)

        returns:
            - pull_logits: raw scores, shape (batch_size,)
            - pull_scores: sigmoid scores between 0 and 1, shape (batch_size,)
        """

        # combine entity's representation and the question vector
        combined = torch.cat(
            [entity_vectors, question_vectors],
            dim=1
        )

        # predict one raw pull score per entity
        pull_logits = self.classifier(combined).squeeze(-1)
        pull_scores = torch.sigmoid(pull_logits)

        return pull_logits, pull_scores


class AnswerScorer(nn.Module):


    def __init__(self, embedding_dim=128):
        super().__init__()

        self.embedding_dim = embedding_dim

        # scorer considers both:
        # the entity's contextual graph representation
        # the representation of the question
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )


    def forward(self, entity_vectors, question_vectors):

        # combine each entity representation with the corresponding question representation
        combined = torch.cat(
            [entity_vectors, question_vectors],
            dim=1
        )

        # one raw answer score (logit) per entity
        answer_logits = self.classifier(combined).squeeze(-1)
        answer_scores = torch.sigmoid(answer_logits)

        return answer_logits, answer_scores
