"""This module indicates the detailed configurations of our framework"""

from __future__ import annotations

import json

KEY_NAMES_LIST = [
    "train_file_path",
    "dev_file_path",
    "test_file_path",
    "mesh_path",
    "mesh_filtering",
    "use_title",
    "use_full",
    "time_step",
    "word_embedding_dim",
    "rel_embedding_dim",
    "char_embedding_dim",
    "pos_embedding_dim",
    "encoder_hidden_size",
    "combined_embedding_dim",
    "lstm_hidden_size",
    "in_attn_heads",
    "out_attn_heads",
    "kernel_size",
    "n_filters",
    "max_seq_length",
    "use_attn",
    "elmo_hidden_size",
    "distance_embedding_dim",
    "entity_hidden_size",
    "max_distance",
    "drop_out",
    "ner_classes",
    "relation_classes",
    "use_ner",
    "lr",
    "batch_size",
    "gradient_clipping",
    "gradient_accumalation",
    "word2vec_path",
    "saved_folder_path",
    "model_type",
    "use_char",
    "use_pos",
    "use_word",
    "use_state",
    "distance_thresh",
    "checkpoint_path",
    "epochs"
]
VALUE_TYPES_DICT = {
    "train_file_path": str,
    "dev_file_path": str,
    "test_file_path": str,
    "mesh_path": str,
    "use_title": bool,
    "mesh_filtering": bool,
    "use_full": bool,
    "time_step": int,
    "word_embedding_dim": int,
    "rel_embedding_dim": int,
    "char_embedding_dim": int,
    "pos_embedding_dim": int,
    "encoder_hidden_size": int,
    "lstm_hidden_size": int,
    "combined_embedding_dim": int,
    "in_attn_heads": int,
    "out_attn_heads": int,
    "kernel_size": int,
    "n_filters": int,
    "max_seq_length": int,
    "use_attn": bool,
    "entity_hidden_size": int,
    "elmo_hidden_size": int,
    "distance_embedding_dim": int,
    "max_distance": int,
    "drop_out": float,
    "ner_classes": int,
    "relation_classes": int,
    "use_ner": bool,
    "lr": float,
    "batch_size": int,
    "gradient_clipping": int,
    "gradient_accumalation": int,
    "word2vec_path": str,
    "saved_folder_path": str,
    "model_type": str,
    "use_char": bool,
    "use_pos": bool,
    "use_word": bool,
    "use_state": bool,
    "distance_thresh": int,
    "checkpoint_path": str,
    "epochs":int
}


class CDRConfig:

    """The CDR Configuration class"""

    chemical_string = "Chemical"
    disease_string = "Disease"
    adjacency_rel = "node"
    root_rel = "root"

    @staticmethod
    def from_json_file(json_file_path: str) -> CDRConfig:
        """load the our method\'s configurations from a json config file

        Args:
            json_file_path (str): path to the json config file

        Returns:
            CDRConfig: an instance of class CDRConfig
        """
        with open(json_file_path) as f_json:
            json_data = json.load(f_json)
            CDRConfig.validate_json_data(json_data)
            config = CDRConfig()
            for attr, value in json_data.items():
                setattr(config, attr, value)
            return config

    @staticmethod
    def validate_json_data(json_data: dict) -> None:
        """validate the json data

        Args:
            json_data (dict): the dictionary that contains param, value pairs after loading the json data.

        Raises:
            Exception: there are some compulsory params which were not defined.
            Exception: contain any param name which not in KEY_NAMES_LIST
            Exception: param type doesn't match. eg given: int and expected: str.
        """
        if len(json_data) < len(KEY_NAMES_LIST):
            missing_params = [key for key in KEY_NAMES_LIST if key not in list(json_data.keys())]
            raise Exception(f"params: {missing_params} must be defined")
        for key, value in json_data.items():
            if key not in KEY_NAMES_LIST:
                raise Exception(f"all config params must be in the pre-defined list: {KEY_NAMES_LIST}")
            if not isinstance(value, VALUE_TYPES_DICT[key]):
                raise Exception(f"Param's type not match. given:{type(value)}, expected:{VALUE_TYPES_DICT[key]}")
