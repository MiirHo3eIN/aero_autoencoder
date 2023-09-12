
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchinfo import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def layer(in_ch, out_ch, T):
    if T: x = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=5, stride=1, padding=2)
    else: x = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=1, padding=2)
    return nn.Sequential(
            x,
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            )


class AE_XL(nn.Module): 
    def __init__(self, ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ch = ch
        self.latentDim = [ch[-1], 800] 
        layers_enc = []
        layers_dec = []

        # print(f"Channels: {ch}")
        for idx, _ in enumerate(ch):
            if idx == len(ch)-1: break
            ridx = len(ch)-idx-1
            # print(f"idx: {idx} | ch: {ch[idx]}")
            # print(f"idx: {idx} | ch-idx: {ridx}")
            layers_enc.append(layer(ch[idx],ch[idx+1], T=False))
            layers_dec.append(layer(ch[ridx], ch[ridx-1], T=True))


        self.encoder = nn.Sequential(*layers_enc)
        self.decoder = nn.Sequential(*layers_dec)

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class AE_d892(AE_XL):
    def __init__(self) -> None:
        ch = [36, 36, 27, 27, 18, 18, 9, 9]
        super().__init__(ch)

arch = {'a619': AE_a619, # simple big model
        '4e36': AE_4e36, # input dropout
        }

def Model(arch_id):
    # print(f"Model({arch_id}), {type(arch_id)}")
    return arch[f"{arch_id}"]()


if __name__ == "__main__":

    input_x = torch.randn(10, 36, 800).to(device)

    print(f"Input: {input_x.shape}")
    dut = Model('4e36')
    summary(dut, input_size = input_x.shape)
    output = dut(input_x)
    print(f"Channels:{dut.ch}")
    print(f"Outputto: {output.shape}")

