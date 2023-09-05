
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchinfo import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Splitter(nn.Module):
    def __init__(self, ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = ch


    def forward(self, x: torch.Tensor)-> list:
        channels = torch.split(x, 1, 1)
        left = []
        right = []
        for sensor, ch in enumerate(channels):
            if sensor in self.channels:
                left.append(ch)
            else:
                right.append(ch)

        return [torch.cat(left, 1), torch.cat(right, 1)]
    
class ReCombiner(nn.Module):
    def __init__(self, left_ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels_left = left_ch


    def forward(self, left: torch.Tensor, right: torch.Tensor)-> torch.Tensor:
        left_ch = torch.split(left, 1, 1)
        right_ch = torch.split(right, 1, 1)
        tensors = []
        left_idx = 0
        right_idx = 0
        for sensor in range(36):
            if sensor in self.channels_left:
                # print(f"Sensor {sensor} -> left")
                tensors.append(left_ch[left_idx])
                left_idx += 1
            else:
                # print(f"Sensor {sensor} -> right")
                tensors.append(right_ch[right_idx])
                right_idx += 1

        return torch.cat(tensors, 1)


class AE_base(nn.Module): 
    def __init__(self, left, left_ch, right_ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left = left
        print(left_ch)
        print(right_ch)
        for idx in range(len(self.left)):
            self.left[idx] -= 1
        self.latentDim = [9, 800] #cf=4

        self.encoder_left = nn.Sequential(
            nn.Conv1d(left_ch[0], left_ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(left_ch[1]),
            nn.ELU(),
            nn.Conv1d(left_ch[1], left_ch[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(left_ch[2]),
            nn.ELU()
        )

        self.decoder_left = nn.Sequential(
            # nn.ConvTranspose1d(left_ch[2], left_ch[1], kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose1d(left_ch[2], left_ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(left_ch[1]),
            nn.ELU(),
            nn.ConvTranspose1d(left_ch[1], left_ch[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(left_ch[0]),
            nn.ELU()
        )

        self.encoder_right = nn.Sequential(
            nn.Conv1d(right_ch[0], right_ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(right_ch[1]),
            nn.ELU(),
            nn.Conv1d(right_ch[1], right_ch[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(right_ch[2]),
            nn.ELU()
        )

        self.decoder_right = nn.Sequential(
            nn.ConvTranspose1d(right_ch[2], right_ch[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(right_ch[1]),
            nn.ELU(),
            nn.ConvTranspose1d(right_ch[1], right_ch[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(right_ch[0]),
            nn.ELU()
        )

        self.splitter = Splitter(self.left)
        self.recombiner = ReCombiner(self.left)

    def getLatentDim(self): return self.latentDim
    def forward(self, x: torch.Tensor):
        l, r = self.splitter(x)
        rec_l = self.decoder_left(self.encoder_left(l))
        rec_r = self.decoder_right(self.encoder_right(r))
        return self.recombiner(rec_l, rec_r)

# Some model form simple models exploration could compress these really well 
class AE_a68f(AE_base):
    def __init__(self) -> None:
        left = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
        left_ch = [len(left), 9, 4]
        right_ch = [36-len(left), 9, 5]
        super().__init__(left, left_ch, right_ch)

# The sensors with positive mean
class AE_d559(AE_base):
    def __init__(self) -> None:
        left = [15, 16, 17]
        left_ch = [len(left), 9, 4]
        right_ch = [36-len(left), 9, 5]
        super().__init__(left, left_ch, right_ch)

# The sensors with positive mean
class AE_5dcd(AE_base):
    def __init__(self) -> None:
        left = [15, 16, 17]
        left_ch = [len(left), 2, 1]
        right_ch = [36-len(left), 18, 8]
        super().__init__(left, left_ch, right_ch)

# The sensors from the middle
class AE_b7e0(AE_base):
    def __init__(self) -> None:
        left = [1, 14, 15, 16, 17, 18]
        left_ch = [len(left), 3, 2]
        right_ch = [36-len(left), 15, 7]
        super().__init__(left, left_ch, right_ch)

arch = {'a68f': AE_a68f, # based on another model
        'd559': AE_d559,
        '5dcd': AE_5dcd,
        'b7e0': AE_b7e0,
        }

def Model(arch_id):
    print(f"Model({arch_id})")
    return arch[arch_id]()


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
    dut = Model('a68f')
    summary(dut, input_size = input_x.shape)
    output = dut(input_x)
    print(f"Outputto: {output.shape}")

