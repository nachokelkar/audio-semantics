from os import makedirs

import numpy as np
import sentencepiece as spm
from gensim.models.word2vec import Word2Vec
from levelwise_model.cluster import Cluster
from fasttext import FastText


class WordToUtteranceMapping:
    def __init__(
            self,
            map_file: str = None
    ):
        self.utterances = {}
        if map_file is not None:
            self.load_mapping(map_file)

    def load_mapping(
            self,
            map_file: str
    ):
        """
        Loads mapping from a file with word and utterance separated by
        a tab-space.
        """
        with open(map_file, "r", encoding='utf-8') as utterance_file:
            for line in utterance_file.readlines():
                if line.strip():
                    key, seq = line.strip().split("\t")

                    if key not in self.utterances:
                        self.utterances[key] = []

                    self.utterances[key].append(seq)

    def save_mapping(
            self,
            map_file: str = None,
            map_dir: str = "data/level_wise/levelX"
    ) -> None:
        if map_file is None:
            map_dir = map_dir.strip("/")
            makedirs(map_dir, exist_ok=True)
            map_file = map_dir + "/utterances.txt"
        with open(map_file, "w+", encoding="utf-8") as map_fp:
            for word in self.utterances:
                for utterance in self.utterances[word]:
                    map_fp.write(word + "\t" + utterance + "\n")

    def update_utterances(
            self,
            sp_model: spm.SentencePieceProcessor,
            clusters: Cluster
    ):
        """
        Updates utterances based on new tokenising and
        clustering.
        """
        new_utterances = {}
        for word in self.utterances:
            new_utterances[word] = []
            for utterance in self.utterances[word]:
                pieces = list(
                    filter(
                        lambda x: x != "▁",
                        sp_model.EncodeAsPieces(utterance)
                    )
                )

                units = [piece.replace("▁", "") for piece in pieces]
                new_utterances[word].append("".join(
                    clusters.word_to_cluster[piece] for piece in units
                ))

        self.utterances = new_utterances

    def get_vectors_from_word_ft(
            self,
            word,
            ft_model: FastText
    ):
        return np.array(
            [
                ft_model.get_sentence_vector(utterance).reshape(1, -1)
                for utterance in self.utterances[word]
            ]
        )

    def get_vectors_from_word(
            self,
            word,
            sp_model: spm.SentencePieceProcessor,
            w2v_model: Word2Vec
    ):
        """
        Gets embeddings of all utterances of a word.
        """
        return np.array(
            [self.get_vector_from_utterance(
                utterance,
                sp_model,
                w2v_model
            ) for utterance in self.utterances[word]]
        )

    def get_vector_from_utterance(
            self,
            utterance,
            sp_model: spm.SentencePieceProcessor,
            w2v_model: Word2Vec
    ):
        """
        Gets the embeddings of the given utterance.
        """
        if utterance in w2v_model.wv.key_to_index.keys():
            return w2v_model.wv[utterance].reshape(1, -1)
        else:
            pieces = list(
                filter(
                    lambda x: x != "▁",
                    sp_model.EncodeAsPieces(utterance)
                )
            )

            units = [piece.replace("▁", "") for piece in pieces]

            vectors = np.array([w2v_model.wv[unit] for unit in units])
            return vectors.mean(axis=0).reshape(1, -1)

    def get_utterance_stats(
            self,
            sp_model: spm.SentencePieceProcessor
    ):
        utterance_lengths = {}
        all_lengths = []
        for word in self.utterances:
            utterance_lengths[word] = []
            for utterance in self.utterances[word]:
                pieces = list(
                    filter(
                        lambda x: x != "▁",
                        sp_model.EncodeAsPieces(utterance)
                    )
                )

                units = [piece.replace("▁", "") for piece in pieces]
                utterance_lengths[word].append(len(units))
                all_lengths.append(len(units))

        return utterance_lengths, all_lengths
