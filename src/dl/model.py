import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
import torchvision.models as models
import torch.optim as optim

''' vision transformer for regression'''
class ViTRegression_V0(nn.Module):
    def __init__(self, num_in_channel, base_model_name = 'vit-base-patch16-224-in21k-finetuned-imagenet'):
        super(ViTRegression_V0, self).__init__()
        self.n_channel = num_in_channel
        # self.vit = VisionTransformer.from_pretrained(base_model_name, num_channels=num_in_channel)
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # self.vit = models.vit_b_16()

        # setup for two class classification
        num_ftrs = self.vit.heads[-1].in_features
        self.vit.heads[-1] = torch.nn.Linear(num_ftrs, 1)
        # method 2: add another layer to 
        
        self.conv1 = nn.Conv2d(num_in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
            
        # self.regression_head = nn.Linear(self.vit.head.in_features, 1)
    def forward(self, x):
        if self.n_channel:
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
        
        x = self.vit(x)
        # x = self.regression_head(x[:, 0])  # Only use the [CLS] token for regression
        return x


''' using resnet, MLP and SDPA'''

class ResNetFNNTranfomerBase(nn.Module):
    def __init__(self, in_channel, num_metadata, num_features, resnet_feature_extractor):
        super(ResNetFNNTranfomerBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        num_resnet_features = self.feature_extract.num_features
        self.fc_metadata = nn.Linear(num_metadata, num_resnet_features)
        self.fc_combined = nn.Linear(num_resnet_features + num_resnet_features, num_features)
        self.fc = nn.Linear(num_resnet_features, num_features)

    def forward(self, img, metadata):
        x_resnet = self.feature_extract(img)
        x_metadata = F.relu(self.fc_metadata(metadata))
        # x_combined = torch.cat((x_resnet, x_metadata), dim=1)
        # x_combined = x_combined.view(x_combined.size(0), -1)
        # x_out = self.fc_combined(x_combined)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x_combined = F.scaled_dot_product_attention(x_metadata,x_resnet,x_resnet)
        x_out = self.fc(x_combined)
        return x_out
    
class ResNetFNNTranfomer_V00(ResNetFNNTranfomerBase):
    def __init__(self, in_channel, num_metadata, num_features, resnet_name='resnet34'):
        super(ResNetFNNTranfomer_V00, self).__init__(in_channel, num_metadata, num_features, ResNetFeatures_V00(in_channel, 0, resnet_name=resnet_name))


'''modified ResNet model for regression
'''
# Base class for modified ResNet model for regression
class ResNetRegressionBase(nn.Module):
    def __init__(self, in_channel, num_features, resnet_feature_extractor):
        super(ResNetRegressionBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        self.fc = nn.Linear(self.feature_extract.num_features, num_features)

    def forward(self, img):
        x = self.feature_extract(img)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Subclasses with different ResNet feature extractors
class ResNetRegression_V00(ResNetRegressionBase):
    def __init__(self, in_channel, num_features, resnet_name='resnet34'):
        super(ResNetRegression_V00, self).__init__(in_channel, num_features, ResNetFeatures_V00(in_channel, 0, resnet_name=resnet_name))

class ResNetRegression_V10(ResNetRegressionBase):
    def __init__(self, in_channel, num_features, resnet_name='resnet34'):
        super(ResNetRegression_V10, self).__init__(in_channel, num_features, ResNetFeatures_V10(in_channel, 0, resnet_name=resnet_name))


'''modified ResNet model for regression from image and metadata
'''
# Base class for combining ResNet and MLP for image and metadata
class ResNetFNNBase(nn.Module):
    def __init__(self, in_channel, num_metadata, num_features, resnet_feature_extractor):
        super(ResNetFNNBase, self).__init__()
        self.feature_extract = resnet_feature_extractor
        num_resnet_features = self.feature_extract.num_features
        self.fc_metadata = nn.Linear(num_metadata, num_resnet_features // 8)
        self.fc_combined = nn.Linear(num_resnet_features + num_resnet_features // 8, num_features)

    def forward(self, img, metadata):
        x_resnet = self.feature_extract(img)
        x_metadata = F.relu(self.fc_metadata(metadata))
        x_combined = torch.cat((x_resnet, x_metadata), dim=1)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_out = self.fc_combined(x_combined)
        return x_out

# Subclasses with different ResNet feature extractors
class ResNetFNN_V00(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_features, resnet_name='resnet34'):
        super(ResNetFNN_V00, self).__init__(in_channel, num_metadata, num_features, ResNetFeatures_V00(in_channel, 0, resnet_name=resnet_name))

class ResNetFNN_V10(ResNetFNNBase):
    def __init__(self, in_channel, num_metadata, num_features, resnet_name='resnet34'):
        super(ResNetFNN_V10, self).__init__(in_channel, num_metadata, num_features, ResNetFeatures_V10(in_channel, 0, resnet_name=resnet_name))


'''# extract features from CNN models''' 
class ResNetFeatures_V00(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V00, self).__init__()\
        
        if resnet_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # method 0: change the first conv1 according to the in_channel
        if in_channel>3:
            weight = resnet.conv1.weight.clone()
            resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)#here 4 indicates 4-channel input
            
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = weight
                for ii in range(3, in_channel):
                    resnet.conv1.weight[:, ii] = resnet.conv1.weight[:, 2]

        self.num_features = num_features if num_features>0 else resnet.fc.in_features 
       
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.resnet.fc(x)
        return x
    

# extract features from CNN models by adding a conv layer to
class ResNetFeatures_V10(nn.Module):
    def __init__(self, in_channel, num_features, resnet_name : str = 'resnet34'):
        super(ResNetFeatures_V10, self).__init__()

        if resnet_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.conv1 = nn.Conv2d(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False) #Input to 3 channel 
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
        
        self.num_features = num_features if num_features>0 else self.resnet.fc.in_features 
    def forward(self, x):
        
        if self.in_channel >3:
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return x



# class NNRegression(nn.Module):
#     def __init__(self, num_features):
#         super(NNRegression, self).__init__()        
#         # self.resnet.fc = nn.Linear(512, 1)
#         self.fc = nn.Linear(num_features, 1)

#     def forward(self, x):        
#         x = self.fc(x)
#         return x


'''in'''

class LSTMRegression(nn.Module):
    def __init__(self, embed_size, hidden_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )

        self.linear = nn.Linear(hidden_size, 1)                     

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=self.device), \
                torch.zeros((1, batch_size, self.hidden_size), device=self.device))
               
    def forward(self, features):
        """ Define the feedforward behavior of the model """
        # Initialize the hidden state
        self.batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
#         print(f'batch_size: {self.batch_size}')
        self.hidden = self.init_hidden(self.batch_size) 
        lstm_out, self.hidden = self.lstm(features, self.hidden) # lstm_out shape : (batch_size, MAX_LABEL_LEN, hidden_size)

        # lstm_out, self.hidden = self.lstm(features.unsqueeze(1), self.hidden) # lstm_out shape : (batch_size, MAX_LABEL_LEN, hidden_size)
        # print(f'lstm_out: {lstm_out.shape}')
#         print(f'hidden: {self.hidden[0].shape}')
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, MAX_LABEL_LEN, vocab_size)
#         print(f'outputs: {outputs.shape}')
        return outputs
    

'''self defined CNN regression model'''
class CNNRegression(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channel,32, kernel_size=3, stride=1, padding= 0, bias=False)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5, stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64,128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(3, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(64*6, 128)
        # self.fc2 = nn.Linear(4098, 1028)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        # print(x.size(1),  x.size(2), x.size(3))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # print(x.size())
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

'''NN regression'''
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size=0, num_classes=0):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        
    def forward(self, x):
        x = torch.flatten(x,1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out

'''encoder'''
class ConvEncoder(nn.Module):
    """ create convolutional layers to extract features
    from input multipe spectral images

    Attributes:
    data : input data to be encoded
    """

    def __init__(self, in_channel):
        super(ConvEncoder,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=64, kernel_size=4,stride=1, padding=0)
        # nn.init.xavier_uniform(self.conv2.weight)
        self.relu= nn.ReLU()

        #Max Pool 1
        # self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        # nn.init.xavier_uniform(self.conv2.weight)
        # self.swish2 = nn.ReLU()

        #Max Pool 2
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        # nn.init.xavier_uniform(self.conv3.weight)
        # self.relu = nn.ReLU()

    def forward(self,x):
        out=self.conv1(x)
        out=self.relu(out)
        size1 = out.size()
        # out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.relu(out)
        size2 = out.size()
        # out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.relu(out)
        return(out)



class DeConvDecoder(nn.Module):
    """ 
    reconstruct image from extracted features

    Attributes:
    features : input data to be encoded
    in_channel: reconstructed channels
    """
    def __init__(self, in_channel):
        super(DeConvDecoder,self).__init__()

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=3)
        # nn.init.xavier_uniform(self.deconv1.weight)
        # self.swish4=nn.ReLU()
        #Max UnPool 1
        # self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=5)
        # nn.init.xavier_uniform(self.deconv2.weight)
        # self.swish5=nn.ReLU()

        #Max UnPool 2
        # self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=64,out_channels=in_channel,kernel_size=4)
        # nn.init.xavier_uniform(self.deconv3.weight)
        # self.swish6=nn.ReLU()
        self.relu= nn.ReLU()

    def forward(self,x):
        out=self.deconv1(x)
        out=self.relu(out)
        # out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.relu(out)
        # out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        # out=self.swish6(out)
        return(out)
    
    