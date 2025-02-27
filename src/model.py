from src.config import IMAGE_WIDTH, IMAGE_HEIGHT

from torch import nn
import torch
from torch.nn import functional as F
class FeatureExtractor(nn.Module):
    def __init__(self, input_channel ):
        """
        Feature Extractor model

        :param input_channel: number of the channel in the image take 2 value 1 , or 3
        """
        super(FeatureExtractor, self).__init__()
        self.input_channel = input_channel
        self.ConvNet = nn.Sequential(
            nn.Conv2d(self.input_channel, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))

    def forward(self, x):
        result = self.ConvNet(x)
        result = result.permute((0 ,3 ,1 ,2))
        batch_size, seq_len, _, _ = result.size()
        result = result.view((batch_size , seq_len , -1))
        return   result

    def get_the_dimension(self ):
        """this function provide the results for the output dimension from CNN model
        """
        x = torch.randn((1 ,self.input_channel ,IMAGE_HEIGHT,IMAGE_WIDTH))
        res = self(x)
        return res.shape


class SequencePredictor(nn.Module):

    def __init__(self, feature_length, output_feature_length=512, hidden_dim=1024, num_category=10):
        """

        :param feature_length:The length of the feature from the CNN layer
        :param output_feature_length:
        :param hidden_dim:
        :param num_category:
        """
        super(SequencePredictor, self).__init__()
        self.Fc = nn.Linear(feature_length, output_feature_length)
        self.Relu = nn.ReLU(inplace=True)
        self.RNN = nn.LSTM(output_feature_length, hidden_dim,
                           batch_first=True, bidirectional=True,)
        self.classification_head = nn.Linear(hidden_dim * 2, num_category)
        #self.activation = nn.Softmax(dim=2)

    def forward(self, x):
        res = self.Fc(x)
        res = self.Relu(res)
        res= self.RNN(res)[0]
        res = self.classification_head (res)
        #res = self.activation(res)
        return res


class OCR_Model(nn.Module):
    def __init__(self,input_channel,output_feature_length = 224,
                 hidden_dim=224, num_category = 10 ):
        super(OCR_Model, self).__init__()
        self.CNN = FeatureExtractor(input_channel=input_channel)
        _ , _ ,feature_length =self.CNN.get_the_dimension()
        self.RNN_and_OUT = SequencePredictor(feature_length= feature_length ,
                                        output_feature_length= output_feature_length ,
                                         hidden_dim = hidden_dim,num_category=num_category)
    def forward(self , x, targets=None):
        bs,_ , _, _  = x.shape
        x =  self.RNN_and_OUT(self.CNN(x))
        x = x.permute(1, 0, 2)
        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32)
            target_lengths = torch.full(size=(bs,), fill_value=targets.size(1), dtype=torch.int32)
            loss = nn.CTCLoss(blank=0)(log_probs, targets, input_lengths, target_lengths)
            return x, loss

        return x, None


