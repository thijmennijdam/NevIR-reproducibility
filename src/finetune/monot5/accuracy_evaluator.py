from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, TripletEvaluator
import logging
import os
import csv
from typing import List
from sentence_transformers.readers import InputExample
import numpy as np
logger = logging.getLogger(__name__)

from external.pygaggle.rerank.base import Query, Text
from external.pygaggle.rerank.transformer import MonoT5

def calc_preferred_rerankers(
    doc1, doc2, q1, q2, model_name="castorini/monot5-base-msmarco-10k", model=None
):
    """
    Input:
        doc1, doc2: strings containing the documents/passages
        q2: string for the query that should be relevant to doc2
        model_name: string containing the type of model to run
        model: the preloaded model, if caching

    Returns:
        A dictionary containing the score (1 if doc2 is ranked higher for q2, otherwise 0).
    """
    if model is None:
        # Model initialization
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        reranker = MonoT5(model=model)
    else:
        reranker = model

    # Prepare passages for reranking
    passages = [[1, doc1], [2, doc2]]
    texts = [
        Text(p[1], {"docid": p[0]}, 0) for p in passages
    ]  # Note: pyserini scores don't matter since T5 will ignore them.

    # Focus only on query q2
    reranked = reranker.rerank(Query(q2), texts)

    # Identify scores for doc1 and doc2
    score_doc1 = next(item.score for item in reranked if item.metadata["docid"] == 1)
    score_doc2 = next(item.score for item in reranked if item.metadata["docid"] == 2)

    # Determine if doc2 is ranked higher
    is_correct = 1 if score_doc2 > score_doc1 else 0

    # Return the result
    results = {
        "q2_scores": [score_doc1, score_doc2],
        "preferred": "doc2" if is_correct else "doc1",
        "score": is_correct,
    }
    return results, model, None



class AccuracyEvaluator(TripletEvaluator):
    """
    Evaluate a model based on a triplet: (q2, doc1, doc2).
    Checks if distance(q2, doc2) < distance(q2, doc1) and computes accuracy.
    """

    def __init__(
        self,
        q1s: List[str],
        q2s: List[str],
        doc1s: List[str],
        doc2s: List[str],
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        model_name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param q1s: List of queries (ignored for accuracy computation).
        :param q2s: List of queries.
        :param doc1s: List of documents.
        :param doc2s: List of documents.
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output.
        :param batch_size: Batch size used to compute embeddings.
        :param show_progress_bar: If true, prints a progress bar.
        :param write_csv: Write results to a CSV file.
        """
        self.q1s = q1s  # Ignored in accuracy computation
        self.q2s = q2s
        self.doc1s = doc1s
        self.doc2s = doc2s
        self.name = name
        self.model_name = model_name

        assert len(self.q1s) == len(self.q2s) == len(self.doc1s) == len(self.doc2s)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "pairwise_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        q1s, q2s, doc1s, doc2s = [], [], [], []

        for example in examples:
            q1s.append(example.texts[0])
            q2s.append(example.texts[1])
            doc1s.append(example.texts[2])
            doc2s.append(example.texts[3])
        return cls(q1s, q2s, doc1s, doc2s, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("PairwiseEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        num_instances = len(self.q2s)
        correct_predictions = 0

        for idx in range(num_instances):
            # Calculate relevance scores using only q2
            results, model, _ = calc_preferred_rerankers(
                self.doc1s[idx], self.doc2s[idx], self.q1s[idx], self.q2s[idx], model_name=self.model_name, model=model
            )
            

            # Check if doc2 is more relevant than doc1
            if results["preferred"] == "doc2":
                correct_predictions += 1

        # Compute accuracy
        accuracy = correct_predictions / num_instances

        logger.info("Accuracy:   \t{:.2f}".format(accuracy * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy
