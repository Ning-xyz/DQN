from src.train import train
from src.eval import evaluate

if __name__ == "__main__":
    type = "train"

    if type == "train" :
        train()
    
    elif type == "eval":
        modelPath = ""
        evaluate(model_path=modelPath)
