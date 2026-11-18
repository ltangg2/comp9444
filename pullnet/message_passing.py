"""
MODEL (graph encoder): give every entity in the question's subgraph a vector
reflecting (a) the question and (b) the facts around it

- model must know how entities are connected
- can be done via a graph NN (message passing)
    - each fact "tells" its two entities about itself and each entity updates its vector accordingly

processing one round (all facts processed at once, in matrix form)
-----------------------------------------------------------------
1. fact vector = tanh(linear([subject_state, relation, object_state, question]))
2. message = role-specific Linear of the fact vector, one for the subject side and one for the object side
3. aggregation = mean of the messages an entity receives (hub entities do not get huge vectors for having many neighbours)
4. update = GRUCell(aggregated message, old state)

- NUM ROUNDS = 2
    - after one round an entity only knows about its direct neighbours
    - 2-hop answer must "see" the topic entity two steps away (3 steps away for 3-hop)
- TOPIC FLAG
    - a learned vector is added to the topic entity so the network knows where reasoning starts
    - i.e. marks which node is the root
- LIMITATION
    - uses a simplified message-passing encoder, not the exact GRAFT-Net/PullNet graph convolution
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

        # one learnable vector per KB entity and per (direction-aware) relation
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # subject + relation + object + question  ->  fact vector
        self.fact_encoder = nn.Linear(embedding_dim * 4, embedding_dim)

        # different transformations for the subject end and the object end
        self.subject_message = nn.Linear(embedding_dim, embedding_dim)
        self.object_message = nn.Linear(embedding_dim, embedding_dim)

        # merge incoming messages into the entity's existing state
        self.entity_updater = nn.GRUCell(embedding_dim, embedding_dim)

        # injects the question into every entity's starting state
        self.question_projection = nn.Linear(embedding_dim, embedding_dim)

        # marks the topic entity
        if use_topic_flag:
            self.topic_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)


    def forward(self, kg, subgraph, question_vector, topic_entity_id=None):
        """
        parameters:
            - kg: full KnowledgeGraph (used to look up fact endpoints/relations)
            - subgraph: QuestionSubgraph (entity IDs + fact IDs to encode)
            - question_vector: shape (embedding_dim,) or (1, embedding_dim)
            - topic_entity_id: required when use_topic_flag is True

        returns:
            - entity_ids: sorted list of KB entity IDs in the subgraph
            - entity_vectors: tensor (len(entity_ids), embedding_dim); row i is the vector of entity_ids[i]
        """
        if len(subgraph.entity_ids) == 0:
            raise ValueError("Question subgraph contains no entities.")
        if self.use_topic_flag and topic_entity_id is None:
            raise ValueError("This encoder uses a topic flag: pass topic_entity_id.")

        device = question_vector.device
        question = question_vector.reshape(1, -1)   # shape (1, embedding_dim)

        # number the entities 0..N-1 so each entity has a row in our tensors
        entity_ids = sorted(subgraph.entity_ids)
        row_of = {}
        for row, entity_id in enumerate(entity_ids):
            row_of[entity_id] = row
        num_entities = len(entity_ids)

        # for each fact, find the rows of its subject and object
        # facts with an endpoint outside the subgraph are skipped
        subject_rows = []
        object_rows = []
        relation_ids = []
        for fact_id in sorted(subgraph.fact_ids):
            fact = kg.get_fact(fact_id)
            if fact.subject_id in row_of and fact.object_id in row_of:
                subject_rows.append(row_of[fact.subject_id])
                object_rows.append(row_of[fact.object_id])
                relation_ids.append(fact.relation_id)

        num_facts = len(relation_ids)
        subject_rows = torch.tensor(subject_rows, dtype=torch.long, device=device)
        object_rows = torch.tensor(object_rows, dtype=torch.long, device=device)
        relation_ids = torch.tensor(relation_ids, dtype=torch.long, device=device)

        # initial entity states
        # entity embedding + question information (+ topic marker)
        entity_id_tensor = torch.tensor(entity_ids, dtype=torch.long, device=device)
        states = (
            self.entity_embeddings(entity_id_tensor)
            + self.question_projection(question)
        )
        if self.use_topic_flag:
            # 1.0 on the topic entity's row, 0.0 on every other row
            is_topic = (entity_id_tensor == topic_entity_id).float().unsqueeze(1)
            states = states + is_topic * self.topic_embedding

        relation_vectors = self.relation_embeddings(relation_ids)
        question_per_fact = question.expand(num_facts, -1)

        # how many messages each entity receives (for the mean)
        ones = torch.ones(num_facts, device=device)
        counts = torch.zeros(num_entities, device=device)
        counts = counts.index_add(0, subject_rows, ones)
        counts = counts.index_add(0, object_rows, ones)
        counts = counts.clamp(min=1).unsqueeze(1)

        # message passing rounds
        for _ in range(self.num_rounds):

            # one vector per fact, built from its subject, relation, object and the question
            fact_input = torch.cat(
                [states[subject_rows], relation_vectors, states[object_rows], question_per_fact],
                dim=1,
            )
            fact_vectors = torch.tanh(self.fact_encoder(fact_input))

            # each fact sends a message to its subject and to its object
            # index_add(0, rows, messages) = for every fact i: total[rows[i]] += messages[i]
            total = torch.zeros_like(states)
            total = total.index_add(0, subject_rows, self.subject_message(fact_vectors))
            total = total.index_add(0, object_rows, self.object_message(fact_vectors))

            # mean of the messages each entity received
            average_message = total / counts

            # update each entity's vector
            states = self.entity_updater(average_message, states)

        return entity_ids, states