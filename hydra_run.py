import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf/config.yaml")
def run_model(cfg: DictConfig) -> None:
    print(cfg.pretty())
    # run(cfg)


if __name__ == "__main__":
    run_model()