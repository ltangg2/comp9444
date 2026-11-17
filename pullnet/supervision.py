"""
supervision for KB-only PullNet

- QA dataset gives questions and correct answers, but not the reasoning paths
    - use shortest paths in the knowledge graph to generate weak training labels (need to know which entities and relations are within the path from Q to A to train the pull-node and relation scorers) for:
        1. Pull nodes: entities that should be expanded at each iteration
        2. Relations: relations that should be retrieved from those entities
    - cache BFS results by (topic entity, maximum hop count), as many questions can share the same topic entity
    (BFS from topic entity and keep shortest path that ends at a gold answer)
    - limitation: generated labels are noisy as the shortest path is not always the intended reasoning path
"""

from collections import defaultdict, deque, OrderedDict

# find shortest paths
def find_shortest_paths(kg, topic_entity_id, max_hops):
    """
    run BFS from topic entity

    for every reachable entity, record:
      - the shortest distance from topic entity
      - all predecessors that reach it at the shortest distance
        - each predecessor stores (parent_entity_id, relation_id)
    
    keep all shortest-distance predecessors in an array to reconstruct all shortest paths
    """

    # distance of 0 as topic entity is the starting point
    distances = {topic_entity_id: 0}

    # predecessors[child] = stores all shortest-path ways of reaching child
    predecessors = defaultdict(list)
    queue = deque([topic_entity_id])

    while queue:
        current = queue.popleft()
        current_distance = distances[current]

        # stop BFS if it will exceed the max number of KB hops
        if current_distance >= max_hops:
            continue

        # retrieve facts connected to the current entity
        for traversal in kg.get_facts_for_entity(current):

            neighbour = traversal.neighbour_id
            relation_id = traversal.relation_id

            new_distance = current_distance + 1

            # first time reaching this neighbour = must be the shortest distance as BFS explores the graph in increasing distance order
            if neighbour not in distances:

                distances[neighbour] = new_distance

                predecessors[neighbour].append(
                    (current, relation_id)
                )

                queue.append(neighbour)

            # if another route reaches the same neighbour at the same shortest distance, keep both to reconstruct all shortest paths
            elif distances[neighbour] == new_distance:

                candidate = (current, relation_id)

                if candidate not in predecessors[neighbour]:
                    predecessors[neighbour].append(candidate)

    predecessors = dict(predecessors)

    return distances, predecessors


# generate labels from shortest paths
def build_supervision_from_bfs(
    kg,
    example,
    max_hops,
    distances,
    predecessors
):
    """
    generate weak supervision for one question using BFS results

    - separate supervision generation function from find_shortest_paths to reuse the same BFS results for multiple Qs with same topic entity
    """

    topic_name = example["topic_entity"]
    answer_names = example["answers"]

    # convert entity names into integers
    topic_id = kg.entity_id(topic_name)

    answer_entity_ids = {
        kg.entity_id(name)
        for name in answer_names
    }

    # keep gold answers reachable within max_hops
    reachable_answer_ids = {
        answer_id
        for answer_id in answer_entity_ids
        if answer_id in distances
    }

    # store unreachable answers from BFS within the hop limit
    unreachable_answers = [
        name
        for name in answer_names
        if kg.entity_id(name) not in reachable_answer_ids
    ]

    # every list position = to one retrieval iteration
    # E.g. index 0 = iteration 1, index 1 = iteration 2
    pull_node_labels = [
        set() for _ in range(max_hops)
    ]

    # for every iteration, map an entity ID to the positive relation IDs
    relation_labels = [
        defaultdict(set) for _ in range(max_hops)
    ]

    # store all unique entities appearing on a shortest path from topic entity to a gold answer
    path_entity_ids = set()

    # traverse from each reachbale gold answer to reconstruct relations and entities to reach topic entity
    for answer_id in reachable_answer_ids:

        stack = [answer_id]
        visited = set()

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            path_entity_ids.add(current)

            # reached starting entity
            if current == topic_id:
                continue

            for parent_id, relation_id in predecessors.get(current, []):

                parent_distance = distances[parent_id]

                # parent at distance 0 = expanded in iteration 1
                # parent at distance 1 = expanded in iteration 2
                iteration_index = parent_distance

                if iteration_index < max_hops:

                    # label the parent as an entity to expand
                    pull_node_labels[iteration_index].add(parent_id)

                    # label the relation connecting parent to the next entity on shortest path
                    relation_labels[iteration_index][parent_id].add(
                        relation_id
                    )

                stack.append(parent_id)

    relation_labels = [
        dict(labels) for labels in relation_labels
    ]

    return {
        "topic_entity_id": topic_id,
        "answer_entity_ids": answer_entity_ids,
        "reachable_answer_ids": reachable_answer_ids,
        "unreachable_answers": unreachable_answers,
        "pull_node_labels": pull_node_labels,
        "relation_labels": relation_labels,
        "path_entity_ids": path_entity_ids,
        "distances": distances,
    }


# generate supervision for one question
def build_supervision(kg, example, max_hops):
    """
    generate supervision for one QA example
        - run BFS from topic entity
        - use BFS results to generate supervision labels
        - use build_supervision_dataset() to process multiple Qs but reusing BFS results
    """

    topic_id = kg.entity_id(example["topic_entity"])

    distances, predecessors = find_shortest_paths(
        kg,
        topic_id,
        max_hops
    )

    return build_supervision_from_bfs(
        kg,
        example,
        max_hops,
        distances,
        predecessors
    )

# validate supervision labels
def validate_supervision(kg, supervision, max_hops):
    """
    check generated supervision labels are consistent with KG and shortest-path distances
        - returns a list of error messages
        - an empty list = all checks passed
    """

    errors = []

    topic_id = supervision["topic_entity_id"]
    distances = supervision["distances"]
    path_entity_ids = supervision["path_entity_ids"]

    pull_node_labels = supervision["pull_node_labels"]
    relation_labels = supervision["relation_labels"]

    # confirm topic entity = starting point
    if distances.get(topic_id) != 0:
        errors.append("Topic entity does not have distance 0.")

    # confirm every reachable answer is included in the path entities
    for answer_id in supervision["reachable_answer_ids"]:

        if answer_id not in path_entity_ids:
            errors.append(
                f"Reachable answer {answer_id} is missing from path entities."
            )

    # confirm number of label groups = max_hops
    if len(pull_node_labels) != max_hops:
        errors.append("Incorrect number of pull-node label iterations.")

    if len(relation_labels) != max_hops:
        errors.append("Incorrect number of relation label iterations.")

    # confirm labels from each iteration
    for iteration_index in range(max_hops):

        pull_nodes = pull_node_labels[iteration_index]
        labels = relation_labels[iteration_index]

        # every entity with relation labels should also be a pull node
        for entity_id in labels:

            if entity_id not in pull_nodes:
                errors.append(
                    f"Entity {entity_id} has relation labels "
                    f"but is not a pull node in iteration "
                    f"{iteration_index + 1}."
                )

        for entity_id in pull_nodes:

            # entity should be at the distance = the iteration where it is expanded
            expected_distance = iteration_index

            if distances.get(entity_id) != expected_distance:
                errors.append(
                    f"Entity {entity_id} is labelled for iteration "
                    f"{iteration_index + 1}, but its distance is "
                    f"{distances.get(entity_id)}."
                )

            # every pull node should have at least one positive relation
            if entity_id not in labels or not labels[entity_id]:
                errors.append(
                    f"Entity {entity_id} has no positive relation "
                    f"in iteration {iteration_index + 1}."
                )
                continue

            # confirm that each positive relation = an actual traversal from this entity to an entity on a
            # shortest path at the next distance
            valid_traversals = {
                (traversal.relation_id, traversal.neighbour_id)
                for traversal in kg.get_facts_for_entity(entity_id)
            }

            for relation_id in labels[entity_id]:

                matching_path_edge = any(
                    (relation_id, neighbour_id) in valid_traversals
                    and neighbour_id in path_entity_ids
                    and distances.get(neighbour_id) == expected_distance + 1
                    for neighbour_id in path_entity_ids
                )

                if not matching_path_edge:
                    errors.append(
                        f"Relation {relation_id} for entity {entity_id} "
                        f"does not lead to a next-step path entity."
                    )

    return errors


# generate supervision for a dataset
def build_supervision_dataset(
    kg,
    examples,
    max_hops,
    bfs_cache_size=32,
    bfs_cache=None
):
    """
    generate supervision for multiple QA examples
        - cache BFS by (topic entity ID, max_hops) for efficiency
            - Qs with identical topic entities and hop limit can reuse the same distances and predecessors
            - cache is limited to bfs_cache_size entries (i.e. processing a large dataset does not retain every BFS result in memory)
    """

    results = []

    # cache = plain dictionary: (topic entity ID, max_hops) -> BFS result
    # retains order when items were added - "oldest" entry = first key
    # a cache passed in by the caller is reused across calls (e.g. across training chunks / epochs)
    if bfs_cache is None:
        bfs_cache = {}

    bfs_cache_hits = 0
    bfs_runs = 0
    questions_with_unreachable_answers = 0
    total_gold_answers = 0
    total_reachable_answers = 0

    for example in examples:

        # convert topic name to integers.
        topic_id = kg.entity_id(example["topic_entity"])

        cache_key = (topic_id, max_hops)

        if cache_key in bfs_cache:

            # reuse previously calculated shortest paths
            distances, predecessors = bfs_cache[cache_key]

            # move current entry to newest position in the cache
            bfs_cache.move_to_end(cache_key)

            bfs_cache_hits += 1

        else:

            # no cached BFS result exists - calculate path
            distances, predecessors = find_shortest_paths(
                kg,
                topic_id,
                max_hops
            )

            bfs_runs += 1
            bfs_cache[cache_key] = (distances, predecessors)

            # remove oldest entry if the cache is too large
            if len(bfs_cache) > bfs_cache_size:
                bfs_cache.popitem(last=False)

        # generate question-specific labels using the shared BFS result
        supervision = build_supervision_from_bfs(
            kg,
            example,
            max_hops,
            distances,
            predecessors
        )

        results.append(supervision)

        # calculate reachability stats
        total_gold_answers += len(supervision["answer_entity_ids"])
        total_reachable_answers += len(
            supervision["reachable_answer_ids"]
        )

        if supervision["unreachable_answers"]:
            questions_with_unreachable_answers += 1

    # calculate total number of positive pull-node labels at each iteration
    positive_pull_nodes_per_iteration = [
        sum(
            len(result["pull_node_labels"][iteration])
            for result in results
        )
        for iteration in range(max_hops)
    ]

    stats = {
        "num_examples": len(examples),
        "bfs_runs": bfs_runs,
        "bfs_cache_hits": bfs_cache_hits,
        "bfs_cache_size_limit": bfs_cache_size,
        "questions_with_unreachable_answers":
            questions_with_unreachable_answers,
        "total_gold_answers": total_gold_answers,
        "total_reachable_answers": total_reachable_answers,
        "positive_pull_nodes_per_iteration":
            positive_pull_nodes_per_iteration,
    }

    return results, stats