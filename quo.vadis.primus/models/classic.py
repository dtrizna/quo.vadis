import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, 
                vocab_size = 152, 
                embedding_dim = 32,
                filter_sizes = [2, 3, 4, 5],
                num_filters = [128, 128, 128, 128],
                num_classes = 2,
                dropout = 0.5
                ):
        super().__init__()

        # embdding
        self.embedding = nn.Embedding(vocab_size, 
                                  embedding_dim, 
                                  padding_idx=0)
        
        # convolutions
        self.conv1d_list = nn.ModuleList([
                            nn.Conv1d(in_channels=embedding_dim,
                                out_channels=num_filters[i],
                                kernel_size=filter_sizes[i])
                            for i in range(len(filter_sizes))
                            ])

        # Fully-connected layers and Dropout
        self.fc_hidden = nn.Linear(np.sum(num_filters), 128)
        self.fc_output = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

        # Non-linearities
        self.relu = torch.nn.ReLU()
    
    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])
    
    def forward(self, inputs):
        # Get embeddings from `x`. 
        # Output shape: (b, max_len, embed_dim), 
        # torch.Size([1024, 150, 32])
        embedded = self.embedding(inputs).permute(0, 2, 1)
        # .permute() to change sequence of max_len and embed_dim, so shape is:
        # torch.Size([1024, 32, 150])
        # needed for Conv1D
        
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv = [self.conv_and_max_pool(embedded, conv1d) for conv1d in self.conv1d_list]
        
        # USED IN PAPER SOMETHING LIKE THIS?
        #x_norm_list = [nn.LayerNorm(x.shape)(x) for x in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = self.dropout(torch.cat(x_conv, dim=1))
        x_h = self.relu(self.fc_hidden(x_fc))
        out = self.fc_output(x_h)
        
        return out


class Modular(nn.Module):
    def __init__(self, 
                # embedding params
                vocab_size = 152,
                embedding_dim = 32,
                # conv params
                filter_sizes = [2, 3, 4, 5],
                num_filters = [128, 128, 128, 128],
                batch_norm_conv = False,
                # ffnn params
                hidden_neurons = [128],
                batch_norm_ffnn = False,
                dropout = 0.5,
                num_classes = 2
                ):
        super().__init__()
        
        # embdding
        self.embedding = nn.Embedding(vocab_size, 
                                  embedding_dim, 
                                  padding_idx=0)
        
        # convolutions
        self.conv1d_module = nn.ModuleList()
        for i in range(len(filter_sizes)):
                if batch_norm_conv:
                    module = nn.Sequential(
                                nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i]),
                                nn.BatchNorm1d(num_filters[i])
                            )
                else:
                    module = nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i])
                self.conv1d_module.append(module)

        # Fully-connected layers
        conv_out = np.sum(num_filters)
        self.ffnn = []

        for i,h in enumerate(hidden_neurons):
            self.ffnn_block = []
            if i == 0:
                self.ffnn_block.append(nn.Linear(conv_out, h))
            else:
                self.ffnn_block.append(nn.Linear(hidden_neurons[i-1], h))
            
            # add BatchNorm to every layer except last
            if batch_norm_ffnn:# and not i+1 == len(hidden_neurons):
                self.ffnn_block.append(nn.BatchNorm1d(h))
            
            self.ffnn_block.append(nn.ReLU())

            if dropout:
                self.ffnn_block.append(nn.Dropout(dropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnn_block))
        
        self.ffnn = nn.Sequential(*self.ffnn)
        self.fc_output = nn.Linear(hidden_neurons[-1], num_classes)
        self.relu = nn.ReLU()

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])
    
    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        
        x_conv = [self.conv_and_max_pool(embedded, conv1d) for conv1d in self.conv1d_module]

        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fc_output(x_fc)        
        
        return out
