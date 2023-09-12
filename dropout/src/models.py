
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


class AE_a619(AE_XL):
    def __init__(self) -> None:
        ch = [36, 36, 27, 27, 18, 18]
        super().__init__(ch)

class AE_4e36(AE_XL):
    def __init__(self) -> None:
        ch = [36, 18]
        super().__init__(ch)
        self.dropout_p = 0.2
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_d12d(AE_XL):
    def __init__(self) -> None:
        ch = [36, 27, 18]
        super().__init__(ch)
        self.dropout_p = 0.2
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_264d(AE_XL):
    def __init__(self) -> None:
        ch = [36, 36, 27, 18]
        super().__init__(ch)
        self.dropout_p = 0.2
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_2c81(AE_XL):
    def __init__(self) -> None:
        ch = [36, 36, 27, 27, 18]
        super().__init__(ch)
        self.dropout_p = 0.2
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_7f8a(AE_XL):
    def __init__(self) -> None:
        ch = [36, 36, 27, 27, 18, 18]
        super().__init__(ch)
        self.dropout_p = 0.2
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_4f53(AE_XL):
    def __init__(self) -> None:
        ch = [36, 18]
        super().__init__(ch)
        self.dropout_p = 0.4
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_a8e1(AE_XL):
    def __init__(self) -> None:
        ch = [36, 18]
        super().__init__(ch)
        self.dropout_p = 0.6
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

class AE_1b45(AE_XL):
    def __init__(self) -> None:
        ch = [36, 18]
        super().__init__(ch)
        self.dropout_p = 0.8
        self.dropout =  nn.Dropout(self.dropout_p)
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.decoder(self.encoder(x))

arch = {'a619': AE_a619, # simple big model
        '4e36': AE_4e36, # input dropout
        'd12d': AE_d12d, # input dropout
        '264d': AE_264d, # input dropout
        '2c81': AE_2c81, # input dropout
        '7f8a': AE_7f8a, # input dropout
        '4f53': AE_4f53, # input dropout
        'a8e1': AE_a8e1, # input dropout
        '1b45': AE_1b45, # input dropout
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

