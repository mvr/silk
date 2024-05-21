import torch
import numpy as np

class SilkNNUE(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = torch.nn.Embedding(14848, 128)
        self.layer2 = torch.nn.Linear(128, 32, bias=True)
        self.layer3 = torch.nn.Linear(64, 32, bias=True)
        self.layer4 = torch.nn.Linear(64, 1, bias=False)

    def forward(self, x):

        assert(x.shape[-1] == 32)
        x = x[..., :29]

        x = self.embedding(x)  # float32[..., 29, 128]
        x = x.sum(dim=-2)      # float32[..., 128]
        x = torch.relu(x)

        x = self.layer2(x)     # float32[..., 32]
        x = torch.cat((x, -x), dim=-1)
        x = torch.relu(x)      # float32[..., 64]

        x = self.layer3(x)     # float32[..., 32]
        x = torch.cat((x, -x), dim=-1)
        x = torch.relu(x)      # float32[..., 64]

        x = self.layer4(x)     # float32[..., 1]
        return x

    def swizzle(self):

        x = [p.cpu().detach().numpy() for p in self.parameters()]

        assert len(x) == 6
        assert tuple(x[0].shape) == (14848, 128)
        assert tuple(x[1].shape) == (32, 128)
        assert tuple(x[2].shape) == (32,)
        assert tuple(x[3].shape) == (32, 64)
        assert tuple(x[4].shape) == (32,)
        assert tuple(x[5].shape) == (1, 64)

        y0 = x[0]
        y1 = np.array([[x[1][i ^ (j >> 2), j] for j in range(128)] for i in range(32)], dtype=np.float32)
        y3 = np.array([[x[3][i ^ (j >> 2), (j >> 2) ^ ((j & 3) << 4)] for j in range(128)] for i in range(16)], dtype=np.float32)
        yb = np.array([[p for i in range(32) for p in [x[2][i], x[4][i], x[5][0, i], x[5][0, i + 32]]]], dtype=np.float32)

        return np.concatenate([y0, y1, y3, yb])
