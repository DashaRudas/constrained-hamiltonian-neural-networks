from lightning.pytorch.cli import LightningCLI
from pl_trainer import DynamicsModel          

if __name__ == "__main__":
    LightningCLI(DynamicsModel, save_config_kwargs={"overwrite": True})