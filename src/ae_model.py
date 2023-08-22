import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchinfo import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class linear_encoder(nn.Module): 

    """ 
    Implementation of a 1-layer linear encoder for autoencoder.
    """
    def __init__(self, d_input: int, d_latent:int): 
        """ 
        Initialize the linear_encoder class.

        Parameters
        ----------
        d_input : int
            Dimension of the input.
        d_latent : int
            Dimension of the latent space.
        """
        super().__init__()
        self.d_input = input 
        self.d_latent = d_latent 
        self.fc1 = nn.Linear(d_input, d_latent)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        """ 
        Propagate the input through the linear_encoder class. 

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        x = self.fc1(x)
        x = F.relu(x)
        return x
    


class linear_decoder(nn.Module):
    """
    Implementing the 1-layer linear decoder for autoencoder.
    """
    def __init__(self, d_latent:int ,d_output:int): 

        """ 
        Initialize the linear_decoder class.

        Parameters
        ----------
        d_latent : int
            Dimension of the latent space.
        d_output : int
            Dimension of the output.
        """

        super().__init__()
        self.d_latent = d_latent
        self.d_output = d_output

        self.fc1 = nn.Linear(d_latent, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ 
        Propagate the input through the linear_decoder class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        
        x = self.fc1(x)
        return x



#class linear_autoencoder(nn.Module):
class AE_6424(nn.Module):

    """
    Implementing the 1-layer linear autoencoder.
    """

    def __init__(self, d_input: int, d_latent:int): 
        """ 
        Initialize the linear_autoencoder class.
        
        Parameters
        ---------- 
        d_input : int
            Dimension of the input.
        d_latent : int
            Dimension of the latent space.
        self.encoder__ : nn.Module 
            Encoder module.
        self.decoder__ : nn.Module
            Decoder module.
        """
        
        super().__init__()
        self.d_input = input 
        self.d_latent = d_latent 
        self.encoder__ = linear_encoder(d_input, d_latent)
        self.decoder__ = linear_decoder(d_latent, d_input)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ 
        Propagate the input through the linear_autoencoder class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        x = self.encoder__(x)
        x = self.decoder__(x)
        return x

###################################################################################################################################################################################
def single_conv1d_block(in_channels: int, out_channel: int, kernel_size: int, stride: int, padding: str, *args, **kwargs) -> nn.Sequential:
    """ 
    Implementing a single convolutional layer with batch normalization and ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. In the input layer this is equal to the number of the sensors used. 
    out_channel : int   
        Number of output channels. Modyfing this parameter effect the spatial relation between the sensors.   
    kernel_size : int
        Kernel size of the convolutional layer. 
    stride : int
        Stride of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    padding : str
        Padding of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    *args :
        Variable length argument list.
    **kwargs :
        Arbitrary keyword arguments.
    """

    return nn.Sequential( 
            nn.Conv1d(in_channels, out_channel, kernel_size, stride, padding, *args, **kwargs), 
            nn.BatchNorm1d(out_channel), 
            nn.ELU()
        )


def single_trans_conv1d_block(in_channels: int, out_channel: int, kernel_size: int, stride: int, padding: int, output_padding: int, *args, **kwargs) -> nn.Sequential:
    """ 
    Implementing a single Trans convolutional layer with batch normalization and ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  
    out_channel : int   
        Number of output channels.   
    kernel_size : int
        Kernel size of the convolutional layer. 
    stride : int
        Stride of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    padding : str
        Padding of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    *args :
        Variable length argument list.
    **kwargs :
        Arbitrary keyword arguments.
    """
    return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channel, kernel_size, stride, padding, output_padding, *args, **kwargs),
            nn.BatchNorm1d(out_channel),
            nn.ELU()
        )

class ConvBlock(nn.Module): 
    """ 
    Implementing a convolutional block with residual connections and downsampling. 

    The ConVBlock is inspired by the architecture in
    "High Fidelity Neural Audio Compression" by Alexandre Defossez et al. (2023)
    
    """
    def __init__(self, c_in: int, c_out: int, kernel_size_residual: int, kernel_size_down_sampling: int,stride_in: int, strid_down_sampling: int): 
        
        """
        Initialize the ConvBlock class.

        Parameters
        ----------
        c_in : int 
            Number of input channels.
        c_out : int
            Number of output channels.
        kernel_size_residual : int
            Kernel size of the residual block in the backbone of the block.
        kernel_size_down_sampling : int
            Kernel size of the downsampling block
        stride_in : int
            Stride of the residual block in the backbone of the block.
        strid_down_sampling : int
            Stride of the downsampling block.
        padding : str   
            padding of the convolutional layers.
        """    
        
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size_residual
        self.stride = stride_in
        self.kernel_down_sampling = kernel_size_down_sampling
        self.stride_down_sampling = strid_down_sampling

        self.backbone = nn.Sequential(
            single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = self.kernel_size, stride = self.stride, padding= 1),
            single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = self.kernel_size, stride = self.stride, padding= 1),
        )
        
        self.downsampling = nn.Conv1d(in_channels = self.c_in, out_channels = c_out, kernel_size = self.kernel_down_sampling, stride = self.stride_down_sampling, padding = 3)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the ConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        
        res = x
        x = self.backbone(x)
        x = x + res
        x = self.downsampling(x)
        
        return x




class trans_conv_block(nn.Module):
    """ 
    Implementing a Transose convolutional block with residual connections and Upsampling for the Decoder part. 

    The ConVBlock is inspired by the architecture in
    "High Fidelity Neural Audio Compression" by Alexandre Defossez et al. (2023)
    
    """
    
    def __init__(self, c_in: int, c_out: int, kernel_size: int, kernel_size_up_sampling: int, stride_residual: int, stride_up_sampling: int, padding: int, output_padding: int, *args, **kwargs): 
        
        """ 
        Initialize the Transpose ConvBlock class.

        Parammeter 
        ----------

        c_in : int
            Number of input channels.
        c_out : int
            Number of output channels.
        kernel_size : int
            Kernel size of the residual block in the backbone of the block.
        kernel_size_up_sampling : int
            Kernel size of the upsampling block
        stride_residual : int  
            Stride of the residual block in the backbone of the block.
        stride_up_sampling : int
            Stride of the upsampling block.
        padding : str
            padding of the convolutional layers.
        output_padding : int
            Output padding of the convolutional layers.
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.
        """
        
        
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.kernel_size_up_sampling = kernel_size_up_sampling
        self.stride_residual = stride_residual
        self.stride_up_sampling = stride_up_sampling
        self.padding = padding
        self.output_padding = output_padding
         
        self.backbone = nn.Sequential(
            single_trans_conv1d_block(in_channels=self.c_in, out_channel= self.c_in, kernel_size=self.kernel_size, stride=self.stride_residual, padding=self.padding, output_padding=self.output_padding),
            single_trans_conv1d_block(in_channels=self.c_in, out_channel= self.c_in, kernel_size=self.kernel_size, stride=self.stride_residual, padding=self.padding, output_padding=self.output_padding)
        )

        self.upsampling = nn.ConvTranspose1d(in_channels=self.c_in, out_channels= self.c_in, kernel_size=self.kernel_size_up_sampling, stride=self.stride_up_sampling, padding=self.padding, output_padding=self.output_padding)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Transpose ConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        res = x 
        x = self.backbone(x)
        x = x + res 
        x = self.upsampling(x)
        return x




class CNN_encoder(nn.Module):

    """
    CNN Encoder class for the Encoder part of the model with a feature extraction layer, one residual block, and downsampling layer.
    
    This is a simplified model compared to the one presented in the paper.
    """

    def __init__(self, c_in): 
        super().__init__()

        """
        Initialize the CNN Encoder class. 

        Parammeter
        ----------
        c_in : int
            Number of input channels.
        
        self.encoder : nn.Sequential
            Sequential model of the Encoder.
        """
        self.c_in = c_in 

        self.encoder = nn.Sequential(
            single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = 7, stride = 2, padding= 3),  
            ConvBlock(c_in = c_in, c_out = c_in, kernel_size_residual = 3, kernel_size_down_sampling = 7, stride_in = 1, strid_down_sampling = 2),
            single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = 7, stride = 2, padding= 3)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the Encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        x = self.encoder(x)
        return x

class Tiny_CNN_encoder(nn.Module):
    """ 
    Tiny CNN Encoder class for the Encoder part of the model with a residual block and downsampling layer.
    
    This is the most simplified model compared to the one presented in the paper.
    """
    def __init__(self, c_in): 
        """      
        Initialize the Tiny CNN Encoder class.
    
        Parammeter
        ----------
        c_in : int
            Number of input channels.
        self.encoder : nn.Sequential 
            Sequential model of the Encoder.
        """

        super().__init__()
        self.c_in = c_in 

        self.encoder = nn.Sequential(
            single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = 7, stride = 2, padding = 3),  
            ConvBlock(c_in = c_in, c_out = c_in, kernel_size_residual = 3, kernel_size_down_sampling = 7, stride_in = 1, strid_down_sampling = 2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        return x


class CNN_decoder(nn.Module): 
    """
    CNN Decoder class for the Decoder part of the model with an Upsampling, one residual block, and a reconstruction layer.
    
    This is a simplified model compared to the one presented in the paper.
    """
    def __init__(self, c_in, *args, **kwargs) -> None:
        """
        Initialize the CNN Decoder class.
        
        Parammeter
        ----------
        c_in : int
            Number of input channels.
        self.decoder : nn.Sequential    
            Sequential model of the Decoder.
        """
        super().__init__(*args, **kwargs)
        self.c_in = c_in

        self.decoder = nn.Sequential(
            single_trans_conv1d_block(in_channels=self.c_in, out_channel= self.c_in, kernel_size=4, stride=2, padding=1, output_padding=0),
            trans_conv_block(c_in=self.c_in, c_out=self.c_in, kernel_size=3, kernel_size_up_sampling=4, stride_residual=1, stride_up_sampling=2, padding=1, output_padding=0),
            single_trans_conv1d_block(in_channels=c_in, out_channel= self.c_in, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the Decoder.
        """ 
        x = self.decoder(x)
        return x
    

class Tiny_CNN_decoder(nn.Module):
    """ 
    Tiny CNN Decoder class for the Decoder part of the model with a residual block and Upsampling layer.
    
    This is the most simplified model compared to the one presented in the paper.
    """
    def __init__(self, c_in, *args, **kwargs) -> None:
        """

        Initialize the Tiny CNN Decoder class.
        
        C_in : int
            Number of input channels.   
        self.decoder : nn.Sequential
            Sequential model of the Decoder.
        """
        super().__init__(*args, **kwargs)
        self.c_in = c_in

        self.decoder = nn.Sequential(
            trans_conv_block(c_in=self.c_in, c_out=self.c_in, kernel_size=3, kernel_size_up_sampling=4, stride_residual=1, stride_up_sampling=2, padding=1, output_padding=0),
            single_trans_conv1d_block(in_channels=c_in, out_channel= self.c_in, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ 
        Propagate the input through the Decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        x = self.decoder(x)
        return x

class AE_8c49(nn.Module):

    """
    Tiny CNN Autoencoder class for the Autoencoder model with an Encoder and a Decoder.

    """ 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        self.encoder = Tiny_CNN_encoder(c_in = self.c_in)
        self.decoder = Tiny_CNN_decoder(c_in = self.c_in)
	
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE_a61c(nn.Module):

    """
    CNN Autoencoder class for the Autoencoder model with an Encoder and a Decoder.

    """ 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        self.encoder = CNN_encoder(c_in = self.c_in)
        self.decoder = CNN_decoder(c_in = self.c_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class AE_3c83(nn.Module):
    """
    Simple autoencoder that reduces the channels to 18

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.c_out = 36 // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(self.c_in, self.c_out, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.c_out),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.c_out, self.c_in, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.c_in),
            nn.ELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_4f90(nn.Module):
    """
    Simple autoencoder that reduces the channels to 9 (cf = 4)

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=9, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x


class AE_942a(nn.Module):
    """
    Simple autoencoder that reduces the channels to 5 (cf = 7.2)

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU(),

            nn.Conv1d(in_channels=9, out_channels=5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(5),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=5, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=9, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_b686(nn.Module):
    """
    Simple autoencoder that reduces the seq len to 100 (cf = 2) 
 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=36, out_channels=36, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(36),
            nn.ELU(),

        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_3ec9(nn.Module):
    """
    Simple autoencoder that reduces the seq len by a factor of 4 (cf = 4) 
 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(36),
            nn.ELU(),

            nn.Conv1d(in_channels=36, out_channels=36, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=36, out_channels=36, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(36),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=36, out_channels=36, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(36),
            nn.ELU(),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_d6eb(nn.Module):
    """
    Simple autoencoder (cf = 8)

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU(),

            nn.Conv1d(in_channels=9, out_channels=9, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(9),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(9),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=9, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_e7c0(nn.Module):
    """
    Simple autoencoder 

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=9, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

class AE_bded(nn.Module):
    """
    Simple autoencoder 

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=18, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(18),
            nn.ELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=18, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(18),
            nn.ELU(),

            nn.ConvTranspose1d(in_channels=18, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x
class AE_cae4(nn.Module):

    """
    Simple autoencoder

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_in = 36
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=9, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(9),
            nn.ELU(),

            # nn.Conv1d(in_channels=9, out_channels=9, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm1d(9),
            # nn.ELU(),
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=7, stride=2, padding=3, output_padding=1),
            # nn.BatchNorm1d(9),
            # nn.ELU(),

            nn.ConvTranspose1d(in_channels=9, out_channels=36, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(36),
            nn.ELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.encoder(x)
        # print(f"Latent Dimensions: {x.shape}")
        x = self.decoder(x)
        return x

arch = {'a61c': AE_a61c, # CNN with residual stuff
        '8c49': AE_8c49, # Tiny CNN with residual stuff
        '6424': AE_6424, # simple linear
        '3c38': AE_3c83, # simple CNN depth = 1 - reduces chan
        '4f90': AE_4f90, # simple CNN depth = 2 - reduces chan
        '942a': AE_942a, # simple CNN depth = 3 - reduces chan
        'b686': AE_b686, # simple CNN depth = 1 - reduces seq
        'd6eb': AE_d6eb, # simple CNN depth = 3 - reduces seq and chan
        '3ec9': AE_3ec9, # simple CNN depth = 2 - reduces seq
        'e7c0': AE_e7c0, # simple CNN depth = 2 - reduces seq
        'cae4': AE_cae4, # simple CNN depth = 2 - reduces seq
        'bded': AE_bded, # simple CNN depth = 3 - reduces seq
        }

    # tiny_cnn = [    "7547:B8DA:C870:507A",
    #                 "829C:AF16:5D58:E61C",
    #                 "C019:A640:74EF:D675",
    #                 "102E:5B5E:C956:FD77"]
    # AE3c38 = ["D9F2:0A27:942A:1DA0"]
    # AE4f90 = ["4E09:4C41:54E0:9BF9"]
    # AE942A = ["372D:4517:E7D3:34E9"]
def Model(arch_id):
    print(f"Model({arch_id})")
    return arch[arch_id]()


if __name__ == "__main__":

    input_x = torch.randn(10, 36, 200).to(device)

    print(f"Input: {input_x.shape}")
    dut = Model('bded')
    summary(dut, input_size = input_x.shape)
    output = dut(input_x)

    print(f"Outputto: {output.shape}")

