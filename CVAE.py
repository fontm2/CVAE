from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys

from dataloader import DataManager



class GaussianCriterion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, logSigma2, target):    
        #negative LL, so sign is flipped
        #1/2*logSigma2 + 0.5 *(2pi)) + 0.5 * (x - mu)^2/sigma^2
        pixelloss = logSigma2.mul(0.5).add(0.5 * torch.log(torch.FloatTensor([2 * np.pi]))).add(torch.add(target, -1, mu).pow(2)/(torch.exp(logSigma2).mul(2)))
        ctx.save_for_backward(mu, logSigma2, target)
        return torch.FloatTensor([pixelloss.sum()])
    @staticmethod
    def backward(ctx, grad_output):
        #in costfunctions grad_output is equal to one (torch.ones(1))
        mu, logSigma2, target = ctx.saved_variables
        grad_mu = grad_logSigma2 = None
        
        if ctx.needs_input_grad[0]:
            # - (target - mu) / sigma^2  --> (1 / sigma^2 = exp(-log(sigma^2)) )
            grad_mu = (torch.exp(-logSigma2)*(torch.add(target,-1, mu))).mul(-1)
        if ctx.needs_input_grad[1]:
            # 0.5 - 0.5 * (target - mu)^2 * exp(-logSigma2) = 0.5 - 0.5 * (target - mu)^2 / sigma^2
            grad_logSigma2 = (torch.exp(-logSigma2)*(torch.add(target,-1,mu).pow(2)).mul(-0.5)).add(0.5)
        return grad_mu, grad_logSigma2, target

class CVAE_creator(nn.Module):
    def __init__(self, latent_variable_size):
        super(CVAE_creator, self).__init__()
        
        conv_depth1 = 8
        conv_depth2 = 16
        conv_depth3 = 32
        conv_depth4 = 64
        conv_depth5 = 128
        conv_depth6 = 256
        self.conv_depth7 = 768
        self.flatten_size = self.conv_depth7*4*4#for 256x256
        #encoder architecture
        #convolution
        self.encoder_Conv = nn.Sequential()
        self.encoder_Conv.add_module("conv0", nn.Conv2d(3, conv_depth1, 3, stride=1, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_LRelu0", nn.LeakyReLU(0.2, inplace=True))       
        self.encoder_Conv.add_module("conv1", nn.Conv2d(conv_depth1, conv_depth2, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm1", nn.BatchNorm2d(conv_depth2))
        self.encoder_Conv.add_module("enc_LRelu1", nn.LeakyReLU(0.2, inplace=True))
        self.encoder_Conv.add_module("conv2", nn.Conv2d(conv_depth2, conv_depth3, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm2", nn.BatchNorm2d(conv_depth3))
        self.encoder_Conv.add_module("enc_LRelu2", nn.LeakyReLU(0.2, inplace=True))
        self.encoder_Conv.add_module("conv3", nn.Conv2d(conv_depth3, conv_depth4, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm3", nn.BatchNorm2d(conv_depth4))
        self.encoder_Conv.add_module("enc_LRelu3", nn.LeakyReLU(0.2, inplace=True))
        self.encoder_Conv.add_module("conv4", nn.Conv2d(conv_depth4, conv_depth5, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm4", nn.BatchNorm2d(conv_depth5))
        self.encoder_Conv.add_module("enc_LRelu4", nn.LeakyReLU(0.2, inplace=True))      
        self.encoder_Conv.add_module("conv5", nn.Conv2d(conv_depth5, conv_depth6, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm5", nn.BatchNorm2d(conv_depth6))
        self.encoder_Conv.add_module("enc_LRelu5", nn.LeakyReLU(0.2, inplace=True))  
        self.encoder_Conv.add_module("conv6", nn.Conv2d(conv_depth6, self.conv_depth7, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.encoder_Conv.add_module("enc_BatchNorm6", nn.BatchNorm2d(self.conv_depth7))
        self.encoder_Conv.add_module("enc_LRelu6", nn.LeakyReLU(0.2, inplace=True))  
        #creat latent_mu- and latent_logSigma2-layer
        #fully-connected
        self.encoder_FC = nn.Sequential()
        self.encoder_FC.add_module("enc_mu",nn.Linear(self.flatten_size, latent_variable_size, bias = False)) 
        self.encoder_FC.add_module("enc_logSigma2",nn.Linear(self.flatten_size, latent_variable_size, bias = False))
                      
        #decoder architecture
        #fully-connected
        self.decoder_FC = nn.Sequential()
        self.decoder_FC.add_module("dec_EntryToConv", nn.Linear(latent_variable_size, self.flatten_size))   
        self.decoder_FC.add_module("dec_EntryToConv_Relu",  nn.ReLU(inplace=True)) 
        #deconvolution
        self.decoder_Conv = nn.Sequential()
        self.decoder_Conv.add_module("dec_Deconv1", nn.ConvTranspose2d(self.conv_depth7, conv_depth6, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm1", nn.BatchNorm2d(conv_depth6))
        self.decoder_Conv.add_module("dec_LRelu1", nn.ReLU(inplace=True))
        self.decoder_Conv.add_module("dec_Deconv2", nn.ConvTranspose2d(conv_depth6, conv_depth5, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm2", nn.BatchNorm2d(conv_depth5))
        self.decoder_Conv.add_module("dec_LRelu2", nn.ReLU(inplace=True))
        self.decoder_Conv.add_module("dec_Deconv3", nn.ConvTranspose2d(conv_depth5, conv_depth4, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm3", nn.BatchNorm2d(conv_depth4))
        self.decoder_Conv.add_module("dec_LRelu3", nn.ReLU(inplace=True))
        self.decoder_Conv.add_module("dec_Deconv4", nn.ConvTranspose2d(conv_depth4, conv_depth3, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm4", nn.BatchNorm2d(conv_depth3))
        self.decoder_Conv.add_module("dec_LRelu4", nn.ReLU(inplace=True))
        self.decoder_Conv.add_module("dec_Deconv5", nn.ConvTranspose2d(conv_depth3, conv_depth2, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm5", nn.BatchNorm2d(conv_depth2))
        self.decoder_Conv.add_module("dec_LRelu5", nn.ReLU(inplace=True))
        self.decoder_Conv.add_module("dec_Deconv6", nn.ConvTranspose2d(conv_depth2, conv_depth1, 4, stride=2, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Conv.add_module("dec_BatchNorm6", nn.BatchNorm2d(conv_depth1))
        self.decoder_Conv.add_module("dec_LRelu6", nn.ReLU(inplace=True))
        #gaussian output
        self.decoder_Out = nn.Sequential()
        self.decoder_Out.add_module("dec_mu", nn.ConvTranspose2d(conv_depth1, 3, 3, stride=1, padding=1, dilation=1, groups=1, bias=False))
        self.decoder_Out.add_module("dec_logSigma2", nn.ConvTranspose2d(conv_depth1, 3, 3, stride=1, padding=1, dilation=1, groups=1, bias=False))
        

    def encode(self, minibatch_input):
        conv_output = self.encoder_Conv(minibatch_input)
        conv_output_flat = conv_output.view(-1, self.flatten_size)#flatten 
        letent_mu = self.encoder_FC.enc_mu(conv_output_flat)  
        latent_logSigma2 = self.encoder_FC.enc_logSigma2(conv_output_flat)  
        return letent_mu, latent_logSigma2

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        fc_Output = self.decoder_FC(z)
        inputToConv = fc_Output.view(-1, self.conv_depth7, 4, 4)                                  
        deconv_output = self.decoder_Conv(inputToConv)        
        output_mu = self.decoder_Out.dec_mu(deconv_output)  
        output_logSigma2 = self.decoder_Out.dec_logSigma2(deconv_output)             
        return output_mu, output_logSigma2

    def forward(self, minibatch_input):
        latent_mu, latent_logSigma2 = self.encode(minibatch_input)
        z = self.reparameterize(latent_mu, latent_logSigma2)
        output_mu, output_logSigma2 = self.decode(z)
        return output_mu, output_logSigma2, latent_mu, latent_logSigma2



def loss_function(output_mu, output_logSigma2, minibatch_input, latent_mu, latent_logSigma2):
    gaussianCriterion = GaussianCriterion.apply
    gaussianLoss = gaussianCriterion(output_mu, output_logSigma2, minibatch_input)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + latent_logSigma2 - latent_mu.pow(2) - latent_logSigma2.exp())
    # Normalise by same number of elements as in reconstruction
    #KLD /= args.batch_size * 784
    return gaussianLoss + KLD

def train(epoch):
    CVAE.train()
    train_loss = 0
    global train_set
    #shuffle on every epoche
    train_set = mymanager.shuffle_withoutLabels(train_set, replace=False)
    for batch_index in range(0, train_set.size()[0]//batch_size):
        startindex = batch_index * batch_size
        train_minibatch = Variable(train_set[startindex:startindex+batch_size,:,:,:])
        if on_cuda:
            train_minibatch = train_minibatch.cuda()
        optimizer.zero_grad()
        output_mu, output_logSigma2, latent_mu, latent_logSigma2 = CVAE(train_minibatch)
        loss = loss_function(output_mu, output_logSigma2, train_minibatch, latent_mu, latent_logSigma2)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        #print("Train: batch_index: ", batch_index, " Loss: ", loss)
    train_loss /= train_set.size()[0]
    print('====> Train set loss: {:.4f} at Epoch: {:}'.format(train_loss, epoch))    
    return output_mu, output_logSigma2, latent_mu, latent_logSigma2, train_loss

def test(epoch):
    CVAE.eval()
    test_loss = 0
    global test_set
    #shuffle on every epoche
    test_set = mymanager.shuffle_withoutLabels(test_set, replace=False)
    for batch_index in range(0, test_set.size()[0]//batch_size):
        startindex = batch_index * batch_size
        test_minibatch = Variable(test_set[startindex:startindex+batch_size,:,:,:])
        if on_cuda:
            test_minibatch = test_minibatch.cuda()      
        output_mu, output_logSigma2, latent_mu, latent_logSigma2 = CVAE(test_minibatch)
        loss = loss_function(output_mu, output_logSigma2, test_minibatch, latent_mu, latent_logSigma2)
        test_loss += loss.data[0]      
        #print("Test: batch_index: ", batch_index, " Loss: ", loss)
    test_loss /= test_set.size()[0]
    print('====> Test set loss: {:.4f} at Epoch: {:}'.format(test_loss, epoch))    
    return test_loss
 
    
    
    
image_path = 'data/processed'
mask_path = 'data/ISIC-2017_Training_Part1_GroundTruth'
label_file = 'data/ISIC-2017_Training_Part3_GroundTruth.csv'

mymanager = DataManager(image_path, mask_path, label_file)
#getting only sebor images
images = mymanager.get_melanoma(as_tensor=True, normalize = False)
#shuffle before splitting into test and train set
images = mymanager.shuffle_withoutLabels(images, replace=False)
#split into train and test set
train_set, test_set = mymanager.datasplit_withoutLabels(images, train_size=0.7)


seed = 1
batch_size = 5
epochs = 400
on_cuda = False
lr=1e-4
#global parameters for early stopping ar indicated with ES_variablename
ES_Refepoches = 5#if the test error does not decrease over this nomber of epoches, do not contiunue with training
ES_epocheCounter = 0
ES_minimal_test_set_error = sys.maxsize

torch.manual_seed(seed)
if on_cuda:
    torch.cuda.manual_seed(seed)   
    
CVAE = CVAE_creator(64)
if on_cuda:
    CVAE.cuda()
    
optimizer = optim.Adam(CVAE.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    output_mu, output_logSigma2, latent_mu, latent_logSigma2, train_loss = train(epoch)
    test_loss = test(epoch)
    #early stopping
    if(test_loss < ES_minimal_test_set_error):
        ES_minimal_test_set_error = test_loss
        torch.save(CVAE.state_dict(), 'CVAE_State_Melanoma22042018.pt')
        #torch.save(optimizer.state_dict(), 'Adam_State.pt')
        ES_epocheCounter = 0
    else:
        ES_epocheCounter += 1
        if(ES_epocheCounter>=ES_Refepoches):
            print("exit via early stoppint")
            break   
