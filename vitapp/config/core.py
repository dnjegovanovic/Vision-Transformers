import sys
from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel
from strictyaml import YAML, load
from yaml.loader import FullLoader

import vitapp

# Project Directories
PACKAGE_ROOT = Path(vitapp.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    save_file: str


class ViT(BaseModel):
    ViT: Dict


class Config(BaseModel):
    """Master config object."""

    model_conf: ViT
    app_config: AppConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as stream:
        try:
            # Converts yaml document to python object
            parsed_config = yaml.load(stream, Loader=FullLoader)
            return parsed_config
        except yaml.YAMLError as e:
            print(e)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        for k, v in parsed_config.items():
            if k == "ViT":
                parsed_config[k]["patch_size"] = int(parsed_config[k]["patch_size"])
                parsed_config[k]["hidden_size"] = int(parsed_config[k]["hidden_size"])
                parsed_config[k]["num_hidden_layers"] = int(
                    parsed_config[k]["num_hidden_layers"]
                )
                parsed_config[k]["num_attention_heads"] = int(
                    parsed_config[k]["num_attention_heads"]
                )
                parsed_config[k]["intermediate_size"] = int(
                    parsed_config[k]["intermediate_size"]
                )
                parsed_config[k]["hidden_dropout_prob"] = float(
                    parsed_config[k]["hidden_dropout_prob"]
                )
                parsed_config[k]["attention_probs_dropout_prob"] = float(
                    parsed_config[k]["attention_probs_dropout_prob"]
                )
                parsed_config[k]["initializer_range"] = float(
                    parsed_config[k]["initializer_range"]
                )
                parsed_config[k]["image_size"] = int(parsed_config[k]["image_size"])
                parsed_config[k]["num_classes"] = int(parsed_config[k]["num_classes"])
                parsed_config[k]["num_channels"] = int(parsed_config[k]["num_channels"])
                parsed_config[k]["qkv_bias"] = bool(parsed_config[k]["qkv_bias"])
            else:
                Exception("No configuration in config file.")

    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_conf=ViT(**parsed_config),
    )

    return _config


config = create_and_validate_config()
