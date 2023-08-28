import torch 
import torch.nn as nn 
from torchinfo import summary

'''
This file contains the only models for the ablation study
These models have to have the function getLatentDim

'''

######################################################################################################
# Helper functions

def reduceChan(c_in, c_out):
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(c_out),
            nn.ELU()
        )

def reduceChanT(c_in, c_out):
        return nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(c_out),
            nn.ELU()
        )

def halveSeq(ch):
    return nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(ch),
            nn.ELU(),
            )

def halveSeqT(ch):
    return nn.Sequential(
            nn.ConvTranspose1d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ch),
            nn.ELU(),
            )

#######################################################################################################
# latent seq reduced Models

class AE_068f(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 400]
        
        self.encoder = halveSeq(36)

        self.decoder = halveSeqT(36)

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_2ec2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 200]
        
        self.encoder = nn.Sequential(
                halveSeq(36),
                halveSeq(36),
            )

        self.decoder = nn.Sequential(
                halveSeqT(36),
                halveSeqT(36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_11e0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 100]
        
        self.encoder = nn.Sequential(
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
            )

        self.decoder = nn.Sequential(
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_3f20(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 50]
        
        self.encoder = nn.Sequential(
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
            )

        self.decoder = nn.Sequential(
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_38eb(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 25]
        
        self.encoder = nn.Sequential(
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
                halveSeq(36),
            )

        self.decoder = nn.Sequential(
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
                halveSeqT(36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))
#######################################################################################################
# latent channel reduced Models

class AE_4fb9(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [18, 800]
        
        self.encoder = reduceChan(36, 18)

        self.decoder = reduceChanT(18, 36)

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_075b(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [9, 800]
        self.encoder = nn.Sequential(
                reduceChan(36, 18),
                reduceChan(18, 9),
            )

        self.decoder = nn.Sequential(
                reduceChanT(9, 18),
                reduceChanT(18, 36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_cc9f(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [5, 800]
        self.encoder = nn.Sequential(
                reduceChan(36, 18),
                reduceChan(18, 9),
                reduceChan(9, 5),
            )

        self.decoder = nn.Sequential(
                reduceChanT(5, 9),
                reduceChanT(9, 18),
                reduceChanT(18, 36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))


class AE_9df3(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [2, 800]
        self.encoder = nn.Sequential(
                reduceChan(36, 18),
                reduceChan(18, 9),
                reduceChan(9, 5),
                reduceChan(5, 2),
            )

        self.decoder = nn.Sequential(
                reduceChanT(2, 5),
                reduceChanT(5, 9),
                reduceChanT(9, 18),
                reduceChanT(18, 36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))

class AE_5b05(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [1, 800]
        self.encoder = nn.Sequential(
                reduceChan(36, 18),
                reduceChan(18, 9),
                reduceChan(9, 5),
                reduceChan(5, 2),
                reduceChan(2, 1),
            )

        self.decoder = nn.Sequential(
                reduceChanT(1, 2),
                reduceChanT(2, 5),
                reduceChanT(5, 9),
                reduceChanT(9, 18),
                reduceChanT(18, 36),
            )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return self.decoder(self.encoder(x))
####################################################################################################
# Helper stuff

arch = {'068f': AE_068f, # CNN reducing seq by 2x
        '2ec2': AE_2ec2, # CNN reducing seq by 4x
        '11e0': AE_11e0, # CNN reducing seq by 8x
        '3f20': AE_3f20, # CNN reducing seq by 16x
        '38eb': AE_38eb, # CNN reducing seq by 32x
        '4fb9': AE_4fb9, # CNN reducing channels by 2x
        '075b': AE_075b, # CNN reducing channels by 4x
        'cc9f': AE_cc9f, # CNN reducing channels by 7x
        '9df3': AE_9df3, # CNN reducing channels by 18x
        '5b05': AE_5b05, # CNN reducing channels by 36x
        }

def Model(arch_id):
    print(f"Model({arch_id}) was loaded")
    return arch[arch_id]()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_x = torch.randn(10, 36, 800).to(device)

    print(f"Input: {input_x.shape}")

    dut = Model('4fb9')
    summary(dut, input_size = input_x.shape)
    output = dut(input_x)
    total_params = sum(p.numel() for p in dut.parameters())

    print(f"Outputto: {output.shape}")
    print(f"latent Dim: {dut.getLatentDim()}")
    print(f"Params: {total_params}")
