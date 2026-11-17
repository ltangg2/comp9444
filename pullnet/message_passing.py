# TO UPDATE COMMENTS

"""
MODEL (graph encoder): give every entity in the question's subgraph a vector that
reflects (a) the question and (b) the facts around it.

Why this exists
---------------
Answering "which movies were directed by the director of [X]?" needs more than
the question text: the model must know how entities are CONNECTED. A graph
neural network (message passing) does this. Each fact "tells" its two entities
about itself, and each entity updates its vector from what it hears.

How one round works (all facts processed at once, in matrix form)
-----------------------------------------------------------------
  1. Fact vector   = tanh( Linear([subject_state, relation, object_state, question]) )
  2. Message       = a role-specific Linear of the fact vector, one for the
                     subject side and one for the object side.
  3. Aggregation   = MEAN of the messages an entity receives (so hub entities
                     do not get huge vectors just for having many neighbours).
  4. Update        = GRUCell(aggregated message, old state)

Two design choices added for 2-hop questions
--------------------------------------------
  * NUM ROUNDS = 2. After one round an entity only knows about its direct
    neighbours. A 2-hop answer must "see" the topic entity two steps away, so
    information has to travel two steps. Weights are shared across rounds.
  * TOPIC FLAG. A learned vector is added to the topic entity so the network
    knows where reasoning starts. Without it, nothing marks which node is the root.

Speed
-----
The first version looped over facts in Python (about 20 tiny PyTorch calls PER FACT,
thousands of facts per question). This version does the same maths with a few
large tensor operations (gather -> Linear -> index_add), which is how GNNs are
normally written and is what makes GPU training worthwhile.

Limitation to mention in the report: this is a simplified message-passing
encoder, not the exact GRAFT-Net/PullNet graph convolution.
"""

import torch
import torch.nn as nn


class QuestionGraphEncoder(nn.Module):

    def __init__(
        self,
        num_entities,
        num_relations,
        embedding_dim=128,
        num_rounds=2,
        use_topic_flag=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_rounds = num_rounds
        self.use_topic_flag = use_topic_flag

        # One learnable vector per KB entity and per (direction-aware) relation.
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # subject + relation + object + question  ->  fact vector
        self.fact_encoder = nn.Linear(embedding_dim * 4, embedding_dim)

        # Different transformations for the subject end and the object end.
        self.subject_message = nn.Linear(embedding_dim, embedding_dim)
        self.object_message = nn.Linear(embedding_dim, embedding_dim)

        # Merge incoming messages into the entity's existing state.
        self.entity_updater = nn.GRUCell(embedding_dim, embedding_dim)

        # Injects the question into every entity's starting state.
        self.question_projection = nn.Linear(embedding_dim, embedding_dim)

        # Marks the topic entity (only created when enabled so that state_dicts
        # of the two variants are not confused).
        if use_topic_flag:
            self.topic_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)

    def forward(self, kg, subgraph, question_vector, topic_entity_id=None):
        """
        Args:
            kg: full KnowledgeGraph (used to look up fact endpoints/relations).
            subgraph: QuestionSubgraph (entity IDs + fact IDs to encode).
            question_vector: shape (embedding_dim,) or (1, embedding_dim).
            topic_entity_id: required when use_topic_flag is True.

        Returns:
            entity_ids:     sorted list of KB entity IDs in the subgraph.
            entity_vectors: tensor (len(entity_ids), embedding_dim); row i is
                            the vector of entity_ids[i].
        """
        if not subgraph.entity_ids:
            raise ValueError("Question subgraph contains no entities.")
        if self.use_topic_flag and topic_entity_id is None:
            raise ValueError("This encoder uses a topic flag: pass topic_entity_id.")

        device = question_vector.device
        question = question_vector.reshape(1, -1)  # (1, D) for broadcasting

        # ---- 1. Index the entities of this subgraph ------------------------
        entity_ids = sorted(subgraph.entity_ids)
        entity_id_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
        num_entities = len(entity_ids)

        # ---- 2. Look up the facts' endpoints and relations -----------------
        # kg.fact_tensors() holds (subject, relation, object) for ALL facts;
        # we gather just the facts in this subgraph.
        all_subjects, all_relations, all_objects = kg.fact_tensors()
        fact_id_tensor = torch.tensor(sorted(subgraph.fact_ids), dtype=torch.long)
        subjects = all_subjects[fact_id_tensor].to(device)
        relations = all_relations[fact_id_tensor].to(device)
        objects = all_objects[fact_id_tensor].to(device)

        # Convert KB entity IDs to row positions (0..N-1) using the sorted ID
        # list; searchsorted + equality check also tells us which facts have
        # both endpoints inside the subgraph (others are ignored).
        last = num_entities - 1
        subject_pos = torch.searchsorted(entity_id_tensor, subjects).clamp(max=last)
        object_pos = torch.searchsorted(entity_id_tensor, objects).clamp(max=last)
        keep = (
            (entity_id_tensor[subject_pos] == subjects)
            & (entity_id_tensor[object_pos] == objects)
        )
        subject_pos, object_pos, relations = (
            subject_pos[keep], object_pos[keep], relations[keep]
        )

        # ---- 3. Initial entity states --------------------------------------
        # entity embedding + question information (+ topic marker)
        states = (
            self.entity_embeddings(entity_id_tensor)
            + self.question_projection(question)
        )
        if self.use_topic_flag:
            is_topic = (entity_id_tensor == topic_entity_id).float().unsqueeze(1)
            states = states + is_topic * self.topic_embedding

        relation_vectors = self.relation_embeddings(relations)  # same every round
        question_per_fact = question.expand(len(relations), -1)

        # How many messages each entity receives (for the mean). Same every round.
        counts = torch.zeros(num_entities, device=device)
        counts = counts.index_add(0, subject_pos, torch.ones_like(subject_pos, dtype=counts.dtype))
        counts = counts.index_add(0, object_pos, torch.ones_like(object_pos, dtype=counts.dtype))
        counts = counts.clamp(min=1).unsqueeze(1)

        # ---- 4. Message passing rounds -------------------------------------
        for _ in range(self.num_rounds):

            fact_input = torch.cat(
                [states[subject_pos], relation_vectors, states[object_pos], question_per_fact],
                dim=1,
            )
            fact_vectors = torch.tanh(self.fact_encoder(fact_input))  # (F, D)

            # Each fact sends a message to its subject and to its object.
            aggregated = torch.zeros_like(states)
            aggregated = aggregated.index_add(0, subject_pos, self.subject_message(fact_vectors))
            aggregated = aggregated.index_add(0, object_pos, self.object_message(fact_vectors))
            aggregated = aggregated / counts  # mean

            states = self.entity_updater(aggregated, states)

        return entity_ids, states
