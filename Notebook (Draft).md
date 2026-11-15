[Introduction, Motivation and/or Problem Statement]

As Natural Language Processing (NLP) models continue to develop throughout the evolution of artificial intelligence, 
it has become a key area of research in the fields of automation that prescribes applicable use cases in Recommender 
Systems, Digital Assistants and Customer Service systems.

Our purpose for Project 003 is to train an AI model on a large knowledge graph of movie facts 
so that it is able to answer natural language questions.

Two properties make project 003 nuanced and motivating:
1. As per NLP, test questions are not linear and uniform, this challenges the model to react dynamically to them.
2. Multi-hop reasoning - Each extra hop chains together another relation in the knowledge graph, and sequential hops 
compounds the difficulty for the model to reason.

We need to first analyse and understand what exactly the MetaQA knowledge base contains, and also what it lacks, 
which will influence our design of the base model’s architecture, and the complex model by extension as well.

[Data Sources or RL Tasks]

MetaQA (MoviE Text Audio QA) is a benchmark of more than 400,000 movie related questions created by Zhang et al. 
It is released under a Creative Commons public license.

Knowledge base: All questions are generated from the knowledge base “kb.txt”. The knowledge base contains one 
relationship per line in the form [node|edge|node]

Question description: The 1 hop questions come from WikiMovies, with ambiguous entities removed. The 2-hop and 3-hop 
questions are generated from the same knowledge base using 21 (2-hop) and 15 (3-hop) question types with 10 text 
templates each.

Question structure: Each question line is in the form of [question<TAB>answer]
For multi answered questions, the formatt is extended [question<TAB>answer1|answer2|...]
Example: [what movies are about [ginger rogers]	Top Hat|Kitty Foyle|The Barkleys of Broadway]

[Exploratory Analysis of Data or RL Tasks]

In analysing the dataset, we looked at the MetaQA knowledge base and its question files.
The knowledge base [kb.txt] contains 134,741 lines written as [node|edge|node].
For example - [Kismet|directed_by|William Dieterie].
This makes the knowledge base a directed, labelled multigraph with 
43,234 nodes, of which 16,427 are movies, and 9 relation types exist.

There exists 9 relationship types: 
[directed_by|written_by|starred_actors|release_year|in_language|has_genre|has_tags|has_imdb_rating|has_imdb_votes]

The nodes are written as plain strings with no IDs or types.
They fall into eight implicit categorie:
[movies|people|years|genres|languages|tags|rating_labels|vote_labels]

Every edge points from a movie to one of its attributes, and nothing points back. 
The spread of edges and nodes are not even, we specifically identified a major skew in the data where language, 
imdb ratings and votes are not sufficiently provided, which likely will impact the performance of our trained NLP 
across these areas.

[1-hop], [2-hop] and [3-hop] questions each split into train, dev and test, with about [96,000], [119,000] and [114,000] 
training questions respectively listed in the README.txt file. 

|       |  1-hop |   2-hop |   3-hop |
|-------|-------:|--------:|--------:|
| Train | 96,106 | 118,980 | 114,196 |
| Dev   |  9,992 |  14,872 |  14,274 |
| Test  |  9,947 |  14,872 |  14,274 |

We can identify The training data is roughly x10 times larger than dev and test.
Each line in the training files contains a question with the topic entity in square brackets, followed by one 
or more answers. Example: [what movies are about [ginger rogers]	Top Hat|Kitty Foyle|The Barkleys of Broadway]

Each hop also contains different question types, with the type being named after the path the question 
followed.

Hop 1

| #  | Question type         |
|----|-----------------------|
| 1  | `movie_to_year`       |
| 2  | `movie_to_director`   |
| 3  | `movie_to_actor`      |
| 4  | `movie_to_writer`     |
| 5  | `movie_to_genre`      |
| 6  | `writer_to_movie`     |
| 7  | `actor_to_movie`      |
| 8  | `movie_to_tags`       |
| 9  | `director_to_movie`   |
| 10 | `tag_to_movie`        |
| 11 | `movie_to_language`   |
| 12 | `movie_to_imdbrating` |
| 13 | `movie_to_imdbvotes`  |

Hop 2

| #  | Question type                   |
|----|---------------------------------|
| 1  | `actor_to_movie_to_director`    |
| 2  | `actor_to_movie_to_writer`      |
| 3  | `actor_to_movie_to_actor`       |
| 4  | `actor_to_movie_to_year`        |
| 5  | `actor_to_movie_to_genre`       |
| 6  | `actor_to_movie_to_language`    |
| 7  | `director_to_movie_to_writer`   |
| 8  | `director_to_movie_to_actor`    |
| 9  | `director_to_movie_to_director` |
| 10 | `director_to_movie_to_year`     |
| 11 | `director_to_movie_to_genre`    |
| 12 | `director_to_movie_to_language` |
| 13 | `writer_to_movie_to_director`   |
| 14 | `writer_to_movie_to_actor`      |
| 15 | `writer_to_movie_to_writer`     |
| 16 | `writer_to_movie_to_year`       |
| 17 | `writer_to_movie_to_genre`      |
| 18 | `writer_to_movie_to_language`   |
| 19 | `movie_to_director_to_movie`    |
| 20 | `movie_to_writer_to_movie`      |
| 21 | `movie_to_actor_to_movie`       |

Hop 3

| #  | Question type                            |
|----|------------------------------------------|
| 1  | `movie_to_actor_to_movie_to_director`    |
| 2  | `movie_to_actor_to_movie_to_writer`      |
| 3  | `movie_to_actor_to_movie_to_year`        |
| 4  | `movie_to_actor_to_movie_to_genre`       |
| 5  | `movie_to_actor_to_movie_to_language`    |
| 6  | `movie_to_writer_to_movie_to_director`   |
| 7  | `movie_to_writer_to_movie_to_actor`      |
| 8  | `movie_to_writer_to_movie_to_year`       |
| 9  | `movie_to_writer_to_movie_to_genre`      |
| 10 | `movie_to_writer_to_movie_to_language`   |
| 11 | `movie_to_director_to_movie_to_writer`   |
| 12 | `movie_to_director_to_movie_to_actor`    |
| 13 | `movie_to_director_to_movie_to_year`     |
| 14 | `movie_to_director_to_movie_to_genre`    |
| 15 | `movie_to_director_to_movie_to_language` |


These categories are matched to each question in the qtype files given in MetaQA, and help analyse the data received 
from trained models. For example, the 3rd line in `1-hop/qa_test_qtype.txt` identifies the question type of the 3rd 
question in `1-hop/vanilla/qa_test.txt`, `1-hop/ntm/qa_test.txt` and `1-hop/audio/qa_test.npz`. It is also important to 
mention that although dev and test are smaller than train in size by a magnitude of 10, they still contain the same 
ratio of question types.

For pre-processing, we first mapped each entity to numerical values; this helps speed up the model’s processing times 
compared to using the default char. Additionally, the presented nodes subject to node knowledge base lines are 
one-directional, so a person like WIllian Dieterie has 17 incoming edges and no outgoing ones, and the model 
effectively cannot travel back in the knowledge graph. To account for this, we also added a reverse copy of every 
edge as its own relation, this gives the knowledge base bi-direcitonal edges that helps the LLM traverse through 
the graph.