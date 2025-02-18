from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, TripletEvaluator
import logging
import os
import csv
from typing import List
from sentence_transformers.readers import InputExample
import numpy as np
logger = logging.getLogger(__name__)
from evaluate.dense import calc_preferred_dense

class PairwiseEvaluator(TripletEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
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
        :param q1s: List of queries
        :param q2s: List of queries
        :param doc1s: List of documents
        :param doc2s: List of documents
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.q1s = q1s
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
        self.csv_headers = ["epoch", "steps", "pairwise_accuracy", "accuracy"]
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

        num_instances = len(self.q1s)
        paired_accuracy, accuracy = 0, 0

        accuracy_list = []
        for idx in range(num_instances):
            results, model, _ = calc_preferred_dense(self.doc1s[idx], self.doc2s[idx], self.q1s[idx], self.q2s[idx], model_name=self.model_name, model=model)
            accuracy_list.append(results['score'])

        paired_accuracy = (np.array(accuracy_list) == 1.0).mean()
        accuracy = np.array(accuracy_list).mean()
        

        # dot product # TODO vectorize it someday if needed
        # q1_dot_distance_doc1s = np.einsum('ij,ij->i', embeddings_q1s, embeddings_doc1s)
        # q1_dot_distance_doc2s = np.einsum('ij,ij->i', embeddings_q1s, embeddings_doc2s)
        # q1_dot_distance_diff = q1_dot_distance_doc1s - q1_dot_distance_doc2s
        # correct_q1s_dot_distance = q1_dot_distance_diff < 0

        # q2_dot_distance_doc1s = np.einsum('ij,ij->i', embeddings_q2s, embeddings_doc1s)
        # q2_dot_distance_doc2s = np.einsum('ij,ij->i', embeddings_q2s, embeddings_doc2s)
        # q2_dot_distance_diff = q2_dot_distance_doc2s - q2_dot_distance_doc1s
        # correct_q2s_dot_distance = q2_dot_distance_diff < 0

        # num_correct_dot_triplets += (correct_q1s_dot_distance & correct_q2s_dot_distance).sum()

        logger.info("Paired Accuracy:   \t{:.2f}".format(paired_accuracy * 100))
        logger.info("Overall Accuracy:   \t{:.2f}".format(accuracy * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, paired_accuracy, accuracy])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, paired_accuracy, accuracy])

        return paired_accuracy