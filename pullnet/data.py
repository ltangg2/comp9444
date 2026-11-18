"""
DATA PREPARATION (part 1): load raw MetaQA files

Files in MetaQA:
  kb.txt
      - movie knowledge base (one fact per line):  subject|relation|object
      - e.g.  Kismet|directed_by|William Dieterle

  {1,2,3}-hop/vanilla/qa_{train,dev,test}.txt
      - One question per line:  question<TAB>answer1|answer2|...
      - e.g.  who directed the films starring [Kismet]?<TAB>...
        - the topic entity (where reasoning starts) is wrapped in [square brackets]

this file reads these MetaQA files and returns plain Python objects - does no
modelling - downstream (graph.py, supervision.py, ...) files build on it
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

HOP_COUNTS = (1, 2, 3)
SPLITS = ("train", "dev", "test")


def get_qa_path(data_dir, hop, split):
    """Build the path  <data_dir>/<hop>-hop/vanilla/qa_<split>.txt."""
    if hop not in HOP_COUNTS:
        raise ValueError(f"hop must be one of {HOP_COUNTS}; got {hop}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split}")
    return Path(data_dir) / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"


def load_knowledge_graph(kb_path):
    """
    read kb.txt into a list of (subject, relation, object) string triples

    Entity names are kept EXACTLY as written (case preserved): the KB contains
    distinct entities that differ only by case, e.g. the actor "Ginger Rogers"
    and the tag "ginger rogers". Lowercasing would merge them.
    """
    triples, malformed = [], []
    with open(kb_path, encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            # every line must have exactly 3 non-empty parts
            if len(parts) != 3 or not all(part.strip() for part in parts):
                malformed.append((line_number, line))
                continue
            triples.append(tuple(part.strip() for part in parts))

    if malformed:
        preview = "\n".join(f"line {n}: {line}" for n, line in malformed[:5])
        raise ValueError(f"{len(malformed)} malformed KB lines:\n{preview}")
    return triples


def load_qa_dataset(file_path):
    """
    read one qa_*.txt file into a list of dictionaries:

        {"question": "...[Kismet]...",     # raw text, brackets kept
         "topic_entity": "Kismet",          # text inside the first [...]
         "answers": ["A", "B", ...]}        # every correct answer
    """
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue

            # Q and A are separated by a tab
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed QA line {line_number} in {file_path}: "
                    "expected a tab between question and answers."
                )
            question, answer_text = parts

            # find topic entity (i.e. text inside [])
            topics = re.findall(r"\[(.*?)\]", question)
            if not topics:
                raise ValueError(
                    f"No [topic entity] on line {line_number}: {question}"
                )

            answers = [a.strip() for a in answer_text.split("|") if a.strip()]
            data.append({
                "question": question.strip(),
                "topic_entity": topics[0].strip(),
                "answers": answers,
            })
    return data


def load_all_datasets(data_dir, hops=HOP_COUNTS):
    """
    load requested hop counts (default: all three) for every split

    returns datasets[hop][split] (i.e. list of question dictionaries)
    pass hops=(2,) to avoid loading unnecessary 1-hop and 3-hop files
    """
    datasets = {}
    for hop in hops:
        datasets[hop] = {}
        for split in SPLITS:
            path = get_qa_path(data_dir, hop, split)
            if not path.exists():
                raise FileNotFoundError(f"Missing QA dataset: {path}")
            datasets[hop][split] = load_qa_dataset(path)
    return datasets