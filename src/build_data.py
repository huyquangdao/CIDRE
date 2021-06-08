from config.cdr_config import CDRConfig
from tqdm import tqdm
from dataset.cdr_dataset import CDRDataset
from corpus.cdr_corpus import CDRCorpus
import argparse
import pickle
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str, help="path to the config.json file", type=str)
    parser.add_argument("--type", default=str, help="type of dataset", type=str)

    args = parser.parse_args()
    config_file_path = "data/config.json"
    config = CDRConfig.from_json_file(config_file_path)
    corpus = CDRCorpus(config)

    # if you still dont have the vocabs for the dataset. You need to call this method firstly.
    print("Preparing all vocabs .....")
    corpus.prepare_all_vocabs(config.saved_folder_path, config.model_type)

    if args.type == "train":
        print("Preparing training data ....")
        corpus.prepare_features_for_one_dataset(
            config.train_file_path, config.model_type, config.saved_folder_path, "train"
        )
    elif args.type == "test":
        print("Preparing testing data ....")
        corpus.prepare_features_for_one_dataset(
            config.test_file_path, config.model_type, config.saved_folder_path, "test"
        )
    elif args.type == "dev":
        print("Preparing development data ....")
        corpus.prepare_features_for_one_dataset(
            config.dev_file_path, config.model_type, config.saved_folder_path, "dev"
        )
    elif args.type == "full":
        print("Preparing all data ....")
        corpus.prepare_features_for_one_dataset(
            config.train_file_path, config.model_type, config.saved_folder_path, "train"
        )
        corpus.prepare_features_for_one_dataset(
            config.test_file_path, config.model_type, config.saved_folder_path, "test"
        )
        corpus.prepare_features_for_one_dataset(
            config.dev_file_path, config.model_type, config.saved_folder_path, "dev"
        )
