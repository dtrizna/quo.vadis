import sys
import logging
import argparse
import pickle

import torch

sys.path.append("..") # repository root
sys.path.append(".")
from models import Emulation, Filepath

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")

    parser.add_argument("--bytes", type=str, help="Pickle serialized file with list of preserved bytes used for training")
    parser.add_argument("--path-model", type=str, help="Path to .toch file with pretrained FilePath model weights")
    parser.add_argument("--apis", type=str, help="Pickle serialized file with list of preserved API calls used for training")
    parser.add_argument("--emu-model", type=str, help="Path to .toch file with pretrained Emulation model weights")
    
    args = parser.parse_args()
    
    if args.path_model:
        if args.bytes:
            logging.warning(f" [*] Loading frequent bytes from {args.bytes}")
            with open(args.bytes, "rb") as f:
                keep_bytes = pickle.load(f)
        else:
            logging.warning(f" [*] You need to provide preserved bytes pickle object to evaluate path")
            sys.exit(1)

        quovadis = Filepath(keep_bytes, "cpu", state_dict=args.path_model)

        # EXAMPLE
        example_path = r"C:\windows\temp\kernel32.exe"
        a,b = quovadis.evaluate_path(example_path)
        print(a,b)
    
    if args.emu_model:
        if args.apis:
            logging.warning(f" [*] Loading preserved API calls from {args.apis}")
            with open(args.apis, "rb") as f:
                api_calls_preserved = pickle.load(f)
            # added labels: 0 - padding; 1 - rare API: therefore range(2, +2)
            apimap = dict(zip(api_calls_preserved, range(2, len(api_calls_preserved)+2)))
        else:
            logging.warning(f" [-] You need to provide preserved API pickle file!")
            sys.exit(1)

        # EXAMPLES
        example_hash = "0a0ab5a01bfed5818d5667b68c87f7308b9beeb2d74610dccf738a932419affd"
        example_pe = "/data/quo.vadis/data/pe.dataset/PeX86Exe/backdoor/0a0ab5a01bfed5818d5667b68c87f7308b9beeb2d74610dccf738a932419affd"

        morbus_certatio = Emulation(apimap, "cpu", state_dict=args.emu_model)    
        a,c = morbus_certatio.evaluate_report(example_hash)
        b,d = morbus_certatio.evaluate_rawpe(example_pe)
        print("\n",example_hash, a, c)
        print(example_pe, b, d)
