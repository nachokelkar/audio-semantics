import logging
from os import makedirs

from gensim.models.word2vec import Word2Vec


class Cluster:
    def __init__(
            self,
            threshold: float = None,
            model: str | Word2Vec = None,
            map_file: str = None,
            log_dir: str = None
    ):
        self.log_dir = log_dir
        self.logger = logging.getLogger("Cluster")
        self.logger.setLevel("DEBUG")
        if log_dir is not None:
            cluster_log_fh = logging.FileHandler(
                self.log_dir + "clusters_log.txt", mode="w"
            )
            log_fh = logging.FileHandler(self.log_dir + "log.txt")
            cluster_log_fh.setLevel("DEBUG")
            log_fh.setLevel("DEBUG")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            cluster_log_fh.setFormatter(formatter)
            log_fh.setFormatter(formatter)
            self.logger.addHandler(cluster_log_fh)
            self.logger.addHandler(log_fh)

        self.cluster_to_words = dict()
        self.word_to_cluster = dict()

        if model is not None and threshold is not None:
            self.create_clusters_from_model(model, threshold)
        elif map_file is not None:
            self.load_mapping(map_file)

    def create_clusters_from_model(
            self,
            model: Word2Vec,
            threshold: float
    ):
        words = list(model.wv.key_to_index.keys())

        cluster_idx = 0  # Counter

        for word in words:
            # Check if word has already been clustered
            if word not in self.word_to_cluster.keys():
                # Create new cluster
                cluster_idx += 1
                while (not chr(0x0020 + cluster_idx).isalpha()) or \
                        len(chr(0x0020 + cluster_idx)) > 1:
                    cluster_idx += 1
                cluster_key = chr(0x0020 + cluster_idx)

                # Add new word to cluster
                self.cluster_to_words[cluster_key] = [word]
                self.word_to_cluster[word] = cluster_key

                # Add all similar words
                for similar_word, score in model.wv.most_similar(
                    word, topn=200
                ):
                    if score > threshold:
                        self.cluster_to_words[cluster_key].append(similar_word)
                        self.word_to_cluster[similar_word] = cluster_key

        self.logger.info(f"Created {len(self.cluster_to_words)} clusters.")

    def load_mapping(
            self,
            map_file: str
    ) -> None:
        with open(map_file, "r", encoding="utf-8") as map_fp:
            for line in map_fp.readlines():
                if line.strip():
                    cluster, words = line.strip().split("\t")
                    words = words.split(",")
                    self.cluster_to_words[cluster] = words
                    for word in words:
                        self.word_to_cluster[word] = cluster

    def save_mapping(
            self,
            map_file: str = None,
            map_dir: str = "data/level_wise/levelX"
    ) -> None:
        if map_file is None:
            map_dir = map_dir.strip("/")
            makedirs(map_dir, exist_ok=True)
            map_file = map_dir + "/clusters.txt"
        print(map_file)
        with open(map_file, "w+", encoding="utf-8") as map_fp:
            for cluster in list(self.cluster_to_words.keys()):
                map_fp.write(
                    cluster + "\t"
                    + ",".join(
                        [word for word in self.cluster_to_words[cluster]]
                    )
                    + "\n"
                )
