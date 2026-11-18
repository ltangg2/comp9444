"""
INFERENCE: the iterative "pull" loop that answers one question

- at test time, there is no BFS teacher, model must find the answer itself
- iteration 1 = start at topic entity, retrieve the best facts
    - encode graph so far
    - PullNodeScorer picks entities worth expanding
- iteration 2 = retrieve facts around those entities and encode again
    - AnswerScorer ranks EVERY entity in the subgraph after last iteration

if the right facts are never retrieved, the answer cannot be ranked - see answer coverage
"""

from retrieval import QuestionSubgraph, retrieve_facts


def run_retrieval_pipeline(
    kg,
    topic_entity_id,
    question_vector,
    relation_scores,
    graph_encoder,
    pull_scorer,
    answer_scorer,
    max_hops=2,
    max_facts_per_entity=10,
    top_k_pull_nodes=10,
    verbose=False,
):
    """
    parameters:
        - question_vector: shape (1, embedding_dim)
        - relation_scores: shape (num_relations,), from RelationScorer
        - max_hops: number of retrieval iterations (2 for 2-hop MetaQA)
        - max_facts_per_entity: facts retrieved each time an entity is expanded
        - top_k_pull_nodes: entities expanded after first iteration

    returns a dictionary with
        - the final subgraph
        - entity IDs/vectors
        - answer logits
        - scores (aligned with entity_ids)
        - per-iteration retrieval_trace
    """

    retrieval_trace = []

    subgraph = QuestionSubgraph()
    subgraph.add_entity(topic_entity_id)

    expanded_entities = set()

    # entities to expand this iteration
    frontier = [topic_entity_id]

    entity_ids, entity_vectors = None, None

    for iteration in range(max_hops):

        if not frontier:
            if verbose:
                print(f"Stopping early at iteration {iteration + 1}: nothing to expand.")
            break

        expanded_entities.update(frontier)

        # pull in the best facts around the frontier entities
        subgraph, new_entity_ids = retrieve_facts(
            kg=kg,
            subgraph=subgraph,
            entity_ids=frontier,
            relation_scores=relation_scores,
            max_facts_per_entity=max_facts_per_entity,
        )

        if verbose:
            print(f"Iteration {iteration + 1}: expanded {len(frontier)}, "
                  f"+{len(new_entity_ids)} entities, "
                  f"{len(subgraph.fact_ids)} facts total")

        # re-encode whole subgraph so vectors reflect the new facts
        entity_ids, entity_vectors = graph_encoder(
            kg, subgraph, question_vector, topic_entity_id=topic_entity_id
        )

        # after the final retrieval there is nothing left to select
        if iteration == max_hops - 1:
            break

        # candidates for the next expansion - entities not expanded yet
        candidate_positions = [
            position
            for position, entity_id in enumerate(entity_ids)
            if entity_id not in expanded_entities
        ]

        if not candidate_positions:
            retrieval_trace.append(
                {"iteration": iteration + 1, "num_candidates": 0, "num_selected": 0}
            )
            frontier = []
            continue

        candidate_vectors = entity_vectors[candidate_positions]
        candidate_questions = question_vector.expand(len(candidate_positions), -1)

        _, pull_scores = pull_scorer(candidate_vectors, candidate_questions)

        # keep top-k highest scoring candidates
        k = min(top_k_pull_nodes, len(candidate_positions))
        selected = pull_scores.detach().topk(k).indices.tolist()

        retrieval_trace.append(
            {"iteration": iteration + 1,
             "num_candidates": len(candidate_positions),
             "num_selected": k}
        )

        frontier = [entity_ids[candidate_positions[p]] for p in selected]

    # loop ends after an encode of the final subgraph (subgraph changes immediately before an encode)
    # those vectorsmcan be reused
    # only encode again in the degenerate max_hops == 0 case
    if entity_vectors is None:
        entity_ids, entity_vectors = graph_encoder(
            kg, subgraph, question_vector, topic_entity_id=topic_entity_id
        )

    # score every entity in the final subgraph as a possible answer
    question_vectors = question_vector.expand(len(entity_ids), -1)
    answer_logits, answer_scores = answer_scorer(entity_vectors, question_vectors)

    return {
        "subgraph": subgraph,
        "entity_ids": entity_ids,
        "entity_vectors": entity_vectors,
        "answer_logits": answer_logits,
        "answer_scores": answer_scores,
        "retrieval_trace": retrieval_trace,
    }
