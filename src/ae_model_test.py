import torch 
from torchinfo import summary

from ae_model import CNN_AE
from ae_model import CNN_encoder, CNN_decoder, ConvBlock, trans_conv_block 


""" 
    This scripts tests the CNN encoder and decoder blocks. 
    The input is random noise. However, the impotant thing is to have the correct shape and sizes. 
    Further to check that the parameters are mathching the expected values. 

"""


def test_ConvBlock(input_x: torch.Tensor): 
    print(input_x.shape)
    conv_block = ConvBlock(c_in = 38, c_out = 64, kernel_size_residual = 3, kernel_size_down_sampling = 7, stride_in = 1, strid_down_sampling = 2)
    summary(conv_block, input_size = input_x.shape)
    output = conv_block(input_x)

    print(output.shape)


def test_trans_conv_block(input_x: torch.Tensor):
    conv_block = trans_conv_block(c_in=152, c_out=76, kernel_size=3, kernel_size_up_sampling=3, stride_residual=1, stride_up_sampling=2, padding=1, output_padding=0)
    summary(conv_block, input_size = input_x.shape)

def test_cnn_encoder(input_x:torch.Tensor): 
    print(input_x.shape)
    cnn_encoder = CNN_encoder(c_in = 38)
    summary(cnn_encoder, input_size = input_x.shape)
    
def test_cnn_decoder(input_x): 
    print(input_x.shape)
    cnn_decoder = CNN_decoder(c_in = 152)
    summary(cnn_decoder, input_size = input_x.shape)



if __name__ == "__main__": 
    input_x = torch.randn(10, 38, 100)
    encoded_x = torch.randn(10, 152, 13)
    
    # Uncomment the type of test you want to run.
    # ------------------------------------------------
    #    test_ConvBlock()
    #    test_cnn_encoder(input_x)
    #    test_trans_conv_block(encoded_x)
    #    test_cnn_decoder(encoded_x)