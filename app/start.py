import sys
sys.path.append("./")
from local_nlp_model import LocalNLPModel


if __name__ == "__main__":
    fip = LocalNLPModel(model_name="together.StableDiffusion")
    fip.start()
