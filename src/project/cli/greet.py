"""CLI entry point for feature extraction."""

import argparse

from utils.logger import init_logging

logger = init_logging(__name__)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract transformer features for CCG probing."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --override model.name=bert-large-uncased",
    )
    return parser.parse_args(argv)


def _load_config(config_path: str, overrides):
    """Load ExperimentConfig from YAML, apply overrides."""
    from omegaconf import OmegaConf

    from project.configs.schema import ExperimentConfig

    cfg_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(cfg_dict, dict)
    for override in overrides:
        key, _, value = override.partition("=")
        keys = key.split(".")
        node = cfg_dict
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Attempt numeric cast
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        node[keys[-1]] = value

    # Map dict -> dataclasses
    from project.configs.schema import (
        DebugConfig,
    )

    debug_cfg = DebugConfig(**cfg_dict.get("debug", {}))

    return ExperimentConfig(
        debug=debug_cfg,
    )


def main(argv=None):
    """Extract features from a transformer model and save to disk."""
    args = _parse_args(argv)
    exp_cfg = _load_config(args.config, args.override)

    message = exp_cfg.debug.message
    logger.info(f"message: {message}")


if __name__ == "__main__":
    main()
