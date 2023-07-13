import logging
from os import makedirs

import numpy as np
import sentencepiece as spm
from gensim.models.word2vec import Word2Vec

from levelwise_model.cluster import Cluster
from levelwise_model.config import Config, SentencePieceConfig, Word2VecConfig
from levelwise_model.test_bench import TestBench
from levelwise_model.utterances import WordToUtteranceMapping


class LevelwiseModel:
    """
    Main level-wise model class.
    """
    def __init__(
            self,
            tag: str = None,
            model_dir: str = "models/",
            data_dir: str = "data/",
            log_dir: str = "logs/",
            results_dir: str = "results/",
    ):
        self.n_levels = 0

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.results_dir = results_dir

        if tag:
            # If there is a tag, it is assumed to be in the
            # default directory
            self.model_dir = self.model_dir + tag
            self.data_dir = self.data_dir + tag
            self.log_dir = self.log_dir + tag
            self.results_dir = self.results_dir + tag

        # Create a logger for debugging and progress tracking
        self.logger = logging.getLogger("LevelwiseModel")
        self.logger.setLevel("DEBUG")
        if log_dir is not None:
            makedirs(self.log_dir, exist_ok=True)
            log_fh = logging.FileHandler(
                self.log_dir.strip("/") + "/log.txt",
                mode="w",
                encoding="utf-8"
            )
            log_fh.setLevel("DEBUG")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            log_fh.setFormatter(formatter)
            self.logger.addHandler(log_fh)

        # Stores paths to models and data
        self.sp_models = []
        self.w2v_models = []
        self.clusters = []
        self.mappings = []

    def _create_level_folders(
            self,
            level: int
    ):
        """
        Function to create directories for each level.
        """
        data_dir = f"{self.data_dir.strip('/')}/level{level}"
        makedirs(data_dir, exist_ok=True)
        self.__curr_data_dir = data_dir
        model_dir = f"{self.model_dir.strip('/')}/level{level}"
        makedirs(model_dir, exist_ok=True)
        self.__curr_model_dir = model_dir
        results_dir = f"{self.results_dir.strip('/')}/level{level}"
        makedirs(results_dir, exist_ok=True)
        self.__curr_results_dir = results_dir

    def get_sentencepiece_from_config(
            self,
            sp_config: SentencePieceConfig,
            input_file: str
    ) -> spm.SentencePieceProcessor:
        """
        Trains a SentencePiece model from a given
        configuration and input file.
        """
        sp_model = spm.SentencePieceProcessor()

        # If model is not specified
        if sp_config.use_model is None:
            model_name = sp_config.get_model_name(
                model_dir=self.__curr_model_dir
            )
            # Train SentencePiece
            self.logger.debug("Training SentencePiece model ...")
            spm.SentencePieceTrainer.train(
                f"--input={input_file} "
                f"--model_type={sp_config.model_type} "
                f"--model_prefix={model_name} "
                f"--vocab_size={sp_config.vocab_size} "
                f"--max_sentence_length={sp_config.max_sentence_length} "
                f"--train_extremely_large_corpus"
            )
            self.logger.info("Created SentencePiece model.")
            self.logger.debug(f"Model path - {model_name}.")

            # Load SentencePiece model
            sp_model.load(f"{model_name}.model")

        else:
            self.logger.info("Using specified SentencePiece model.")
            self.logger.debug(f"Model path - {sp_config.use_model}")

            # Load specified model
            sp_model.load(f"{sp_config.use_model}")

        return sp_model

    def get_word2vec_from_config(
            self,
            w2v_config: Word2VecConfig,
            sentences: list
    ) -> Word2Vec:
        """
        Trains Word2Vec model with given configuration.
        """
        # If W2V model is not specified
        if w2v_config.use_model is None:
            # Train Word2Vec
            self.logger.debug("Training Word2Vec model ...")

            w2v_model_path = w2v_config.get_model_path(
                model_dir=self.__curr_model_dir
            )
            w2v_model = Word2Vec(
                sentences,
                window=w2v_config.window,
                vector_size=w2v_config.vector_size,
                min_count=0,
                workers=4,
                epochs=7
            )
            w2v_model.save(w2v_model_path)
            self.logger.info("Created Word2Vec model.")
            self.logger.debug(f"Model path - {w2v_model_path}.")

        else:
            # Use specified model
            w2v_model = Word2Vec.load(w2v_config.use_model)

        return w2v_model

    def train(
            self,
            input_file: str,
            utterance_file: str,
            n_levels: int = None,
            configs: list[Config] = None,
            test_bench: TestBench = None
    ) -> None:
        """
        Main training function

        Input
        -----
            input_file : str
                Filepath of the input. The file must contain a sentence
                per line, with maximum line length of 5000.
            configs : list
                Arguments for SentencePiece and Word2Vec at each level.
        """
        # Load initial utterances file
        utterances = WordToUtteranceMapping()
        utterances.load_from_file(map_file=utterance_file)

        if n_levels is None:
            if configs is None:
                raise ValueError(
                    "At least one of `n_levels` or `configs` must be used."
                )
            else:
                n_levels = len(configs)

        for level in range(1, n_levels + 1):
            self.logger.info(f"----- STARTING LEVEL {level} -----")

            # Create folder for level files
            self._create_level_folders(level)
            self.logger.info("Created level folders.")

            # Load run config
            config = Config()
            if len(configs) >= level:
                config = configs[level - 1]

            sp_model = self.get_sentencepiece_from_config(
                sp_config=config.sp_config,
                input_file=input_file
            )

            # Convert input file to sentences
            with open(input_file, "r", encoding="utf-8") as corpus_fp:
                corpus = corpus_fp.readlines()

            sentences = []
            for sentence in corpus:
                pieces = list(
                    filter(
                        lambda x: x != "▁",
                        sp_model.EncodeAsPieces(sentence)
                    )
                )

                new_pieces = [piece.replace("▁", "") for piece in pieces]
                sentences.append(new_pieces)
            self.logger.info("Converted input to sentences")
            self.logger.debug(f"Sample sentence - {sentences[0][:10]}")

            # Fetch Word2Vec model
            w2v_model = self.get_word2vec_from_config(
                w2v_config=config.w2v_config,
                sentences=sentences
            )

            # Perform clustering
            cluster = Cluster(
                model=w2v_model,
                threshold=config.cluster_threshold
            )

            # Save next level corpus
            input_file = f"{self.__curr_data_dir}/corpus.txt"
            with open(input_file, "w+", encoding="utf-8") as corpus_fp:
                for line in sentences:
                    corpus_fp.write(
                        "".join(
                            cluster.word_to_cluster[piece] for piece in line
                        ) + "\n"
                    )
            self.logger.info(f"Created level {level} corpus.")

            # Create a function that generates vectors for an input
            # at this level
            def word_vec_fn(word):
                if word in w2v_model.wv.key_to_index.keys():
                    return w2v_model.wv[word].reshape(1, -1)
                else:
                    pieces = list(
                        filter(
                            lambda x: x != "▁",
                            sp_model.EncodeAsPieces(word)
                        )
                    )

                    units = [piece.replace("▁", "") for piece in pieces]

                    vectors = np.array([w2v_model.wv[unit] for unit in units])
                    return vectors.mean(axis=0).reshape(1, -1)

            # Test level
            if test_bench:
                self.logger.debug("Testing layer ...")
                test_bench.run_suite(
                    utterances=utterances,
                    word_vec_fn=word_vec_fn,
                    results_file=self.__curr_results_dir + "/results.txt"
                )
                self.logger.info("Tested layer.")

            # Update utterances
            self.logger.debug("Updating utterances ...")
            utterances.update_utterances(
                sp_model,
                cluster
            )
            self.logger.info("Updated utterances.")

            # Save all files
            cluster.save_mapping(map_dir=self.__curr_data_dir)
            utterances.save_mapping(map_dir=self.__curr_data_dir)
            self.logger.info("Saved cluster and utterance files.")

        self.logger.info("Completed training.")
