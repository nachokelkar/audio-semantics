"""
This module contains the configuration classes
for SentencePiece and Word2Vec
"""


class SentencePieceConfig:
    def __init__(
        self,
        max_sentence_length: int = 5000,
        vocab_size: int = 20000,
        model_type: str = "unigram",
        model_tag: str = "lw",
        use_model: str = None
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.model_tag = model_tag
        self.use_model = use_model

    def get_model_name(
            self,
            model_name: str = None,
            model_dir: str = "models/level_wise/levelX"
    ):
        if model_name is not None:
            return model_name
        return f"{model_dir.strip('/')}/" \
            f"{self.model_type}_vs{self.vocab_size}_{self.model_tag}"


class Word2VecConfig:
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        model_tag: str = "lw",
        use_model: str = None
    ):
        self.vector_size = vector_size
        self.window = window
        self.model_tag = model_tag
        self.use_model = use_model

    def get_model_path(
            self,
            model_name: str = None,
            model_dir: str = "models/level_wise/levelX"
    ):
        if model_name is not None:
            return model_name
        return f"{model_dir.strip('/')}/" \
            f"w2v_vs{self.vector_size}_w{self.window}_{self.model_tag}.model"


class Config:
    def __init__(
            self,
            sp_config: SentencePieceConfig = SentencePieceConfig(),
            w2v_config: Word2VecConfig = Word2VecConfig(),
            cluster_threshold: float = 0.45
    ):
        self.sp_config = sp_config
        self.w2v_config = w2v_config
        self.cluster_threshold = cluster_threshold
