import time
import argparse
import logging

from utils.servers import HTTP_Server

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training filepath NeuralNetwork.")
    
    # model parameters
    parser.add_argument("--model-path", type=str, default="objects/model_50.torch")
    parser.add_argument("--frequent-bytes", type=str, default="objects/keep_bytes.pickle")
    parser.add_argument("--padding-length", type=int, default=150)
    # server parameters
    parser.add_argument("--port", type=int, default=80, help="Port to serve the model via HTTP.")
    parser.add_argument("--address", type=str, default="0.0.0.0", help="Address to bind HTTP server.")
    # auxiliary
    parser.add_argument("--logfile", type=str, help="File to store logging messages.")
    parser.add_argument("--debug", action="store_true", help="Provide with DEBUG level information from packages.")


    args = parser.parse_args()

    # if logfile argument present - log to a file instead of stdout
    level = logging.DEBUG if args.debug else logging.WARNING
    if args.logfile:
        logging.basicConfig(handlers=[logging.FileHandler(args.logfile, 'a', 'utf-8')], level=level)
    else:
        logging.basicConfig(level=level)

    logging.warning(f" [*] {time.ctime()}: Staring server on {args.address}:{args.port}")
    
    # temp config or /tests/PROD_RUN/
    #HTTP_Server(args.address, args.port)
    HTTP_Server(args.address, args.port,
                    #model_path=args.model_path,
                    model_path="tests/PROD_RUN/ep100-optim_adam-lr0.001-l2reg0-dr0.5_arr-ed64-kb150-pl150_model-conv128-bn_cFalse_f_True-ffnn256_128-model.torch",
                    bytes_used="tests/PROD_RUN/keep_bytes_1641410834.pickle",
                    padding_length=150,
                    embedding_dim=64,
                    model_type="modular",
                    batchnorm=True,
                    ffnn = [256, 128]
                )
