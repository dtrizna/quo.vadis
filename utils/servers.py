from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import pickle

import sys
sys.path.append("..")
from models import Modular
from utils.evaluation import evaluate_list


class HTTP_Server:
    def __init__(self, address, port, device="cpu",
                    model_path = "objects/model_50.torch",
                    bytes_used = "objects/keep_bytes.pickle",
                    model_type="classic",
                    padding_length = 150, 
                    embedding_dim = 32,
                    batchnorm = False,
                    ffnn = [128]
                ):
        server_address = (address, port)
        HTTP_RequestHandler.padding_length = padding_length
        
        # provide bytes used during encoding
        with open(bytes_used, "rb") as f:
            self.keep_bytes = pickle.load(f)
        HTTP_RequestHandler.keep_bytes = self.keep_bytes

        # instantiate model to be available from RequestHandler
        if model_type == "classic":
            self.model = Net().to(device)
            model_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(model_dict)
            HTTP_RequestHandler.model = self.model
        elif model_type == "modular":
            self.model = Modular(
                                    vocab_size = len(self.keep_bytes) + 2,
                                    embedding_dim = embedding_dim,
                                    hidden_neurons = ffnn,
                                    batch_norm_ffnn = batchnorm
                                    #filter_sizes = args.filter_sizes,
                                    #num_filters = [args.num_filters] * len(args.filter_sizes),
                                    #dropout=args.dropout
                                ).to(device)
            model_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(model_dict)
            HTTP_RequestHandler.model = self.model
        else:
            raise NotImplementedError

        httpd = HTTPServer(server_address, HTTP_RequestHandler)
        httpd.serve_forever()


class HTTP_RequestHandler(BaseHTTPRequestHandler):
    model = None
    keep_bytes = None
    padding_length = None

    def do_GET(self):
        self.send_error(405, "Use POST to provide filepath(s) to evaluate.")

    def do_POST(self):    
        if self.path == "/predict":
            self.model = self.model
        else:
            self.send_error(404, "Unknown URL. Supported now: /predict")
            return

        content_len = self.headers.get("Content-Length")
        if not content_len:
            self.send_error(404, "Please supply filepath as POST data!")
        else:
            post_body = self.rfile.read(int(content_len)).decode()
            input_list = post_body.strip().split("\n")

            prob_list = evaluate_list(self.model, input_list, self.keep_bytes, padding_length=self.padding_length)
            response_body_list = [f"{input_list[i]: <50}: prob(malware) = {probs[1]:.4f}" for i,probs in enumerate(prob_list)]
            response_body = "\n".join(response_body_list) + "\n"

            self.send_response(200)
            self.end_headers()
            self.wfile.write(response_body.encode("utf-8"))



    
    
    

