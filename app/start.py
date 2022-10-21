import sys
import argparse
sys.path.append("./")
from local_nlp_model import LocalNLPModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--model-name', type=str, default='together.gptj6b', metavar='S',
                        help='trained model path')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S',
                        help='cuda-id (default:0)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='S',
                        help='batch-size for inference (default:8)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    args = parser.parse_args()

    fip = LocalNLPModel(model_name=args.model_name, args=args)
    fip.start()
