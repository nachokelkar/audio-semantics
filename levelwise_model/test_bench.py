from json import dump

import numpy as np
import sentencepiece as spm
from gensim.models.word2vec import Word2Vec
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from levelwise_model.utterances import WordToUtteranceMapping


class TestBench:
    def load_scores(
            self,
            scores_file: str
    ):
        pass

    def score(self):
        pass

    def score_and_save(
            self,
            results_file: str = None
    ):
        pass


class LSTestBench(TestBench):
    def __init__(
            self,
            scores_file: str = None
    ) -> None:
        self.load_scores(scores_file)

    def load_scores(
            self,
            scores_file: str
    ):
        """
        Loads scores from the LibriSpeech dataset.
        The file consists of rows of the format:
            word1,word2,similarity_score,relationness_score
        """
        self.sim_pairs = []
        self.rel_pairs = []

        with open(scores_file, "r") as pairs_file:
            for line in pairs_file.readlines()[1:]:
                w1, w2, sim, rel = line.strip().split(",")
                if sim:
                    self.sim_pairs.append((w1, w2, float(sim)))
                if rel:
                    self.rel_pairs.append((w1, w2, float(rel)))

    def single_test(
            self,
            pairs: list,
            sp_model: spm.SentencePieceProcessor,
            w2v_model: Word2Vec,
            utterances: WordToUtteranceMapping
    ):
        scores = {
            test_set: {
                method: [] for method in ["min", "max", "avg", "all"]
            } for test_set in ["librispeech", "synthetic"]
        }
        gold_standard = {
            "librispeech": [],
            "synthetic": []
        }
        trials = 0
        errors = 0

        for pair in pairs:
            try:
                w1, w2, rel = pair

                test_set = "librispeech" \
                    if w1.startswith("ls_") \
                    else "synthetic"
                w1.replace("ls_", "").replace("sy_", "")
                w2.replace("ls_", "").replace("sy_", "")

                w1_vectors = utterances.get_vectors_from_word(
                    w1, sp_model, w2v_model
                )
                w2_vectors = utterances.get_vectors_from_word(
                    w2, sp_model, w2v_model
                )

                similarities = [
                    cosine_similarity(i, j)
                    for i in w1_vectors
                    for j in w2_vectors
                ]

                scores[test_set]["min"].append(np.min(similarities))
                scores[test_set]["avg"].append(np.mean(similarities))
                scores[test_set]["max"].append(np.max(similarities))

                gold_standard[test_set].append(rel)
            except Exception as e:
                print(e)
                errors += 1
            trials += 1

        return {
            'score': {
                test_set: {
                    var: pearsonr(
                        scores[test_set][var],
                        gold_standard[test_set]
                    )[0] * 100
                    for var in ['min', 'avg', 'max']
                }
                for test_set in ['librispeech', 'synthetic']
            },
            'errors': errors,
            'trials': trials
        }

    def score(
            self,
            sp_model: spm.SentencePieceProcessor,
            w2v_model: Word2Vec,
            utterances: WordToUtteranceMapping
    ) -> dict:
        # tests = {'sim' : self.sim_pairs, 'rel' : self.rel_pairs}
        tests = {'rel': self.rel_pairs}

        return {
            test: self.single_test(
                tests[test],
                sp_model,
                w2v_model,
                utterances
            ) for test in tests
        }

    def score_and_save(
            self,
            sp_model: spm.SentencePieceProcessor,
            w2v_model: Word2Vec,
            utterances: WordToUtteranceMapping,
            results_file: str = "results/level_wise/levelX"
    ):
        results = self.score(
            sp_model=sp_model,
            w2v_model=w2v_model,
            utterances=utterances
        )

        dump(results, open(results_file, "w+", encoding="utf-8"))

    def ft_single_test(
            self,
            pairs: list,
            ft_model,
            utterances: WordToUtteranceMapping
    ):
        scores = {
            test_set: {
                method: [] for method in ["min", "max", "avg", "all"]
            } for test_set in ["librispeech", "synthetic"]
        }
        gold_standard = {
            "librispeech": [],
            "synthetic": []
        }
        trials = 0
        errors = 0

        for pair in pairs:
            try:
                w1, w2, rel = pair

                test_set = "librispeech" \
                    if w1.startswith("ls_") \
                    else "synthetic"
                w1.replace("ls_", "").replace("sy_", "")
                w2.replace("ls_", "").replace("sy_", "")

                w1_vectors = utterances.get_vectors_from_word_ft(
                    w1, ft_model=ft_model
                )
                w2_vectors = utterances.get_vectors_from_word_ft(
                    w2, ft_model=ft_model
                )

                similarities = [
                    cosine_similarity(i, j)
                    for i in w1_vectors
                    for j in w2_vectors
                ]

                scores[test_set]["min"].append(np.min(similarities))
                scores[test_set]["avg"].append(np.mean(similarities))
                scores[test_set]["max"].append(np.max(similarities))

                gold_standard[test_set].append(rel)
            except Exception as e:
                print(e)
                errors += 1
            trials += 1

        return {
            'score': {
                test_set: {
                    var: pearsonr(
                        scores[test_set][var],
                        gold_standard[test_set]
                    )[0] * 100
                    for var in ['min', 'avg', 'max']
                }
                for test_set in ['librispeech', 'synthetic']
            },
            'errors': errors,
            'trials': trials
        }

    def ft_score(
            self,
            ft_model,
            utterances: WordToUtteranceMapping
    ):
        # tests = {'sim' : self.sim_pairs, 'rel' : self.rel_pairs}
        tests = {'rel': self.rel_pairs}

        return {
            test: self.ft_single_test(
                tests[test],
                ft_model,
                utterances
            ) for test in tests
        }

    def ft_score_and_save(
            self,
            ft_model,
            utterances: WordToUtteranceMapping,
            results_file: str = "results/ft_st"
    ):
        results = self.ft_score(
            ft_model,
            utterances=utterances
        )

        dump(results, open(results_file, "w+", encoding="utf-8"))
