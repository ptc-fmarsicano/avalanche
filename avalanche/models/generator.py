################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

File to place any kind of generative models 
and their respective helper functions.

"""

from abc import abstractmethod
from matplotlib import transforms
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

from avalanche.models.base_model import BaseModel


class Generator(BaseModel):
    """
    A base abstract class for generators
    """

    @abstractmethod
    def generate(self, batch_size=None):
        """
        Lets the generator sample random samples.
        Output is either a single sample or, if provided,
        a batch of samples of size "batch_size" 
        """


###########################
# VARIATIONAL AUTOENCODER #
###########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    '''
    Simple nn.Module to flatten each tensor of a batch of tensors.
    '''

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    '''
    Simple nn.Module to create a multi-layer perceptron 
    with BatchNorm and ReLU activations.

    :param hidden_size: An array indicating the number of neurons in each layer.
    :type hidden_size: int[]
    :param last_activation: Indicates whether to add BatchNorm and ReLU 
                            after the last layer.
    :type last_activation: Boolean
    '''

    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2)
                                            and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    '''
    Encoder part of the VAE, computer the latent represenations of the input.

    :param shape: Shape of the input to the network: (channels, height, width)
    :param latent_dim: Dimension of last hidden layer
    '''

    def __init__(self, shape, latent_dim=128):
        super(Encoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            MLP([400, latent_dim])
                                   )

    def forward(self, x, y=None):
        x = self.encode(x)
        return x


class Decoder(nn.Module):
    '''
    Decoder part of the VAE. Reverses Encoder.

    :param shape: Shape of output: (channels, height, width).
    :param nhid: Dimension of input.
    '''

    def __init__(self, shape, nhid=16):
        super(Decoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.shape = shape
        self.decode = nn.Sequential(
            MLP([nhid, 64, 128, 256, flattened_size], last_activation=False),
            nn.Sigmoid())
        self.invTrans = transforms.Compose([
                                    transforms.Normalize((0.1307,), (0.3081,))
                        ])

    def forward(self, z, y=None):
        if (y is None):
            return self.invTrans(self.decode(z).view(-1, *self.shape))
        else:
            return self.invTrans(self.decode(torch.cat((z, y), dim=1))
                                 .view(-1, *self.shape))


class VAE(nn.Module):
    '''
    Variational autoencoder module: 
    fully-connected and suited for any input shape and type.

    The encoder only computes the latent represenations
    and we have then two possible output heads: 
    One for the usual output distribution and one for classification.
    The latter is an extension the conventional VAE and incorporates
    a classifier into the network.
    More details can be found in: https://arxiv.org/abs/1809.10635
    '''

    def __init__(self, shape, nhid=16, n_classes=10):
        """
        :param shape: Shape of each input sample
        :param nhid: Dimension of latent space of Encoder.
        :param n_classes: Number of classes - 
                        defines classification head's dimension
        """
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, latent_dim=128)
        self.calc_mean = MLP([128, nhid], last_activation=False)
        self.calc_logvar = MLP([128, nhid], last_activation=False)
        self.classification = MLP([128, n_classes], last_activation=False)
        self.decoder = Decoder(shape, nhid)

    def sampling(self, mean, logvar):
        """
        VAE 'reparametrization trick'
        """
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    # Orginial forward of VAE.
    # We modify this to allow for Replay-through-Feedback,
    # see VAEPlugin for details.
    # def forward(self, x):
    #    mean, logvar = self.encoder(x)
    #    z = self.sampling(mean, logvar)
    #    return self.decoder(z), mean, logvar
    def forward(self, x):
        """
        Forward. Computes representations of encoder.
        """
        return self.encoder(x)

    def generate(self, batch_size=None):
        """
        Generate random samples.
        Output is either a single sample if batch_size=None,
        else it is a batch of samples of size "batch_size". 
        """
        z = torch.randn((batch_size, self.dim)).to(
            device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res


# Loss functions    
BCE_loss = nn.BCELoss(reduction="sum")
MSE_loss = nn.MSELoss(reduction="sum")
CE_loss = nn.CrossEntropyLoss()


def VAE_loss(X, X_hat, mean, logvar):
    '''
    Loss function of a VAE using mean squared error for reconstruction loss.
    This is the criterion for VAE training loop.

    :param X: Original input batch.
    :param X_hat: Reconstructed input after subsequent Encoder and Decoder.
    :param mean: mean of the VAE output distribution.
    :param logvar: logvar of the VAE output distribution.
    '''
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence
