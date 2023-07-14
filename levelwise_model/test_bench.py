from json import dump
from string import ascii_letters

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from levelwise_model.utterances import WordToUtteranceMapping


class TestBench:
    """
    Template test bench class.
    """
    def load_scores(
            self,
            scores_file: str
    ):
        """
        Template function to load similarity scores.
        """
        pass

    def run_suite(
            self,
            results_file: str = None
    ):
        """
        Template function to run all tests.
        """
        pass


class LSTestBench(TestBench):
    """
    Test suite for LibriSpeech dataset. This is specific to the
    data from Zerospeech and would require extra processing to be
    extended.
    """
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

    def similarity_score_test(
            self,
            utterances: WordToUtteranceMapping,
            word_vec_fn: callable
    ):
        """
        Uses similarity scores from the given pairs list
        and finds correlation between similarities provided by
        the callable.

        This function is specific for the format the pairs are
        set in.

        Inputs
        ------
            utterances : WordToUtteranceMapping
                To fetch the utterances of each word.
            word_vec_fn : callable
                A function that returns embeddings for a
                string input.
        """
        scores = {
            test_set: []
            for test_set in ["librispeech", "synthetic"]
        }
        gold_standard = {
            "librispeech": [],
            "synthetic": []
        }
        trials = 0
        errors = 0

        for pair in self.rel_pairs:
            try:
                w1, w2, rel = pair

                test_set = "librispeech" \
                    if w1.startswith("ls_") \
                    else "synthetic"

                # Get vectors
                w1_vectors = np.array(
                    [word_vec_fn(utt) for utt in utterances.utterances[w1]]
                )
                w2_vectors = np.array(
                    [word_vec_fn(utt) for utt in utterances.utterances[w2]]
                )

                # Compute similarities between each pair
                similarities = [
                    cosine_similarity(i, j)
                    for i in w1_vectors
                    for j in w2_vectors
                ]

                # Append scores
                scores[test_set].append(np.mean(similarities))
                gold_standard[test_set].append(rel)

            except Exception as e:
                print(e)
                errors += 1
                break

            trials += 1

        # Return the output score
        return {
            'score': {
                test_set: pearsonr(
                        scores[test_set],
                        gold_standard[test_set]
                    )[0] * 100
                for test_set in ['librispeech', 'synthetic']
            },
            'errors': errors,
            'trials': trials
        }

    def same_word_abx_test(
            self,
            utterances: WordToUtteranceMapping,
            word_vec_fn: callable,
            runs_per_ds: int = 700,
            use_noise_for_x: bool = False
    ):
        """
        Performs ABX testing with all the words in the utterance
        vocaulary. For each word, a test is only done if there are
        at least two other word similarities available for it.

        Inputs
        ------
            utterances : WordToUtteranceMapping
                To fetch the utterances of each word.
            word_vec_fn : callable
                A function that returns embeddings for a
                string input.
            use_noise_for_x : bool
                Whether to randomly generate word X. `false`
                uses the word with least similarity with
                word A instead. (default = False)
        """
        test_sets = {
            'librispeech': {
                'preds': 0,
                'total': 0,
                'dataset': utterances.ls_utterances
            },
            'synthetic': {
                'preds': 0,
                'total': 0,
                'dataset': utterances.sy_utterances
            },
            'mixed': {
                'preds': 0,
                'total': 0,
                'dataset': utterances.mixed_utterances
            },
        }

        preds = 0
        total = 0

        for test_set in test_sets:
            ds = test_sets[test_set]['dataset']
            for _ in range(runs_per_ds):
                # Select two random words for A and X
                word_a, word_x = np.random.choice(
                    list(ds.keys()),
                    size=2,
                    replace=False
                )
                utt_a, utt_b = np.random.choice(
                    ds[word_a],
                    size=2,
                    replace=False
                )

                utt_x = np.random.choice(
                    ds[word_x]
                )

                v_a = word_vec_fn(utt_a)
                v_b = word_vec_fn(utt_b)
                v_x = word_vec_fn(utt_x)

                sim_ab = cosine_similarity(v_a, v_b)
                sim_ax = cosine_similarity(v_a, v_x)

                if sim_ab > sim_ax:
                    # If model predicts A and B to be closer
                    # than A and x, it is a success
                    preds += 1
                    test_sets[test_set]['preds'] += 1
                total += 1
                test_sets[test_set]['total'] += 1

        return {
            "Same-word ABX Result": {
                test_set: (
                    test_sets[test_set]['preds'] /
                    test_sets[test_set]['total']
                )
                for test_set in test_sets
            }
        }

    def cross_word_abx_test(
            self,
            utterances: WordToUtteranceMapping,
            word_vec_fn: callable,
            use_noise_for_x: bool = False,
    ):
        """
        Performs ABX testing with all the words in the utterance
        vocaulary. For each word, a test is only done if there are
        at least two other word similarities available for it.

        Inputs
        ------
            utterances : WordToUtteranceMapping
                To fetch the utterances of each word.
            word_vec_fn : callable
                A function that returns embeddings for a
                string input.
            use_noise_for_x : bool, OPTIONAL
                Whether to randomly generate word X. `false`
                uses the word with least similarity with
                word A instead. (default = False)
        """
        ls_preds = 0
        sy_preds = 0
        ls_total = 0
        sy_total = 0
        preds = 0
        total = 0

        def __get_other_word(pair, word):
            """
            Custom function to return the other word given
            a pair
            """
            if pair[0] == word:
                return pair[1]
            return pair[1]

        for word_a in utterances.utterances:
            # Sort words by similarity
            similar_words = sorted(
                # Get only pairs containing word A
                filter(
                    lambda x: x[0] == word_a or x[1] == word_a,
                    self.rel_pairs
                ),
                key=lambda x: x[2],
                reverse=True
            )

            # Get the other words (the one that isn't word A from pair)
            similar_words = list(
                map(
                    lambda x: __get_other_word(x, word_a),
                    similar_words
                )
            )

            # For all utterances of the word
            for utt_a in utterances.utterances[word_a]:
                # Word B is most similar word
                word_b = similar_words[0]  # TODO: Change to sample from top n%
                # Use random utterance
                utt_b = np.random.choice(
                    utterances.utterances[word_b]
                )

                if not use_noise_for_x:
                    # Word X is least similar word
                    word_x = similar_words[-1]
                    # Use random utterance
                    utt_x = np.random.choice(
                        utterances.utterances[word_x]
                    )
                else:
                    word_x = "noise"  # Placeholder word
                    # Generate noise
                    utt_x = np.random.choice(
                        list(ascii_letters),
                        len(utt_a),
                        replace=True
                    )
                    utt_x = "".join(utt_x)

                # TODO: Perform this check before indexing
                if len(set([word_a, word_b, word_x])) == 3:
                    v_a = word_vec_fn(utt_a)
                    v_b = word_vec_fn(utt_b)
                    v_x = word_vec_fn(utt_x)

                    sim_ab = cosine_similarity(v_a, v_b)
                    sim_ax = cosine_similarity(v_a, v_x)

                    if sim_ab > sim_ax:
                        if word_a.startswith("ls_"):
                            ls_preds += 1
                        elif word_a.startswith("sy_"):
                            sy_preds += 1
                        # If model predicts A and B to be closer
                        # than A and x, it is a success
                        preds += 1
                    if word_a.startswith("ls_"):
                        ls_total += 1
                    elif word_a.startswith("sy_"):
                        sy_total += 1
                    total += 1

        return {
            "ABX Result": {
                'librispeech': ls_preds / ls_total,
                'synthetic': sy_preds / sy_total,
                'combined': preds / total
            }
        }

    def run_suite(
            self,
            utterances: WordToUtteranceMapping,
            word_vec_fn: callable,
            results_file: str = None
    ) -> dict:
        """
        Runs all tests and saves to a results file.

        Inputs
        ------
            utterances : WordToUtteranceMapping
                To fetch the utterances of each word.
            word_vec_fn : callable
                A function that returns embeddings for a
                string input.
            results_file : str, OPTIONAL
                File path to save results in. Saves it as a
                JSON. Passing `None` will not save results
                to a file.
        """
        results = {
            "similarities": self.similarity_score_test(
                utterances=utterances,
                word_vec_fn=word_vec_fn
            ),
            "xw-abx-test": self.cross_word_abx_test(
                utterances=utterances,
                word_vec_fn=word_vec_fn,
                use_noise_for_x=False
            ),
            "sw-abx-test": self.same_word_abx_test(
                utterances=utterances,
                word_vec_fn=word_vec_fn,
                use_noise_for_x=False
            )
        }

        if results_file is not None:
            dump(results, open(results_file, "w+", encoding="utf-8"))

        return results
