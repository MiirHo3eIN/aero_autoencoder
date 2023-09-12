
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchinfo import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AE_1L(nn.Module): 
    def __init__(self, ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [ch[-1], 800] 

        self.encoder = nn.Sequential(
            nn.Conv1d(ch[0], ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[1]),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ch[1], ch[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[0]),
            nn.ELU()
        )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

class AE_2L(nn.Module): 
    def __init__(self, ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [ch[-1], 800] 

        self.encoder = nn.Sequential(
            nn.Conv1d(ch[0], ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[1]),
            nn.ELU(),
            nn.Conv1d(ch[1], ch[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[2]),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ch[2], ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[1]),
            nn.ELU(),
            nn.ConvTranspose1d(ch[1], ch[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(ch[0]),
            nn.ELU()
        )

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

class AE_2619(AE_2L):
    def __init__(self) -> None:
        ch = [36, 18, 18]
        super().__init__(ch)

class AE_7579(AE_2L):
    def __init__(self) -> None:
        ch = [36, 27, 18]
        super().__init__(ch)

class AE_56a9(AE_1L):
    def __init__(self) -> None:
        ch = [36, 18]
        super().__init__(ch)

class AE_5c50(nn.Module): 
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latentDim = [36, 800] 

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor): return x



arch = {'2619': AE_2619, # based on another model
        '56a9': AE_56a9,
        '7579': AE_7579,
        '5c50': AE_5c50,
        }

def Model(arch_id):
    # print(f"Model({arch_id}), {type(arch_id)}")
    return arch[f"{arch_id}"]()


if __name__ == "__main__":

    input_x = torch.randn(10, 36, 800).to(device)

    print(f"Input: {input_x.shape}")
    # splitter = Splitter([1, 5, 9]).to(device)
    # z, y = splitter(input_x)
    # print(f"Z: {z.shape}")
    # print(f"Y: {y.shape}")
    # rec = ReCombiner([1, 5, 9]).to(device)
    # w = rec(z, y)
    # print(f"W: {w.shape}")
    dut = Model('2619')
    summary(dut, input_size = input_x.shape)
    output = dut(input_x)
    print(f"Outputto: {output.shape}")

