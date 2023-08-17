from os import makedirs

import sentencepiece as spm

from levelwise_model.cluster import Cluster
from typing import Union


class WordToUtteranceMapping:
    def __init__(
            self,
            utterances: Union[dict, str] = None,
    ):
        self.utterances = {}
        self.ls_utterances = {}
        self.sy_utterances = {}
        self.mixed_utterances = {}
        if isinstance(utterances, dict):
            self.load_from_dict(utterances)
        elif isinstance(utterances, str):
            self.load_from_file(utterances)
        self.split_utterances_by_dataset()

    def load_from_dict(
            self,
            utterances: dict
    ) -> None:
        self.utterances = utterances

    def split_utterances_by_dataset(self) -> None:
        for word in self.utterances:
            if word.startswith("ls_"):
                self.ls_utterances[word] = self.utterances[word]
            if word.startswith("sy_"):
                self.sy_utterances[word] = self.utterances[word]
        for word in self.utterances:
            if word[3:] not in self.mixed_utterances:
                self.mixed_utterances[word[3:]] = []
                if "ls_" + word[3:] in self.ls_utterances:
                    self.mixed_utterances[word[3:]] = \
                        self.mixed_utterances[word[3:]] + \
                        self.ls_utterances["ls_" + word[3:]]
                if "sy_" + word[3:] in self.sy_utterances:
                    self.mixed_utterances[word[3:]] = \
                        self.mixed_utterances[word[3:]] + \
                        self.sy_utterances["sy_" + word[3:]]

    def load_from_file(
            self,
            map_file: str
    ) -> None:
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
