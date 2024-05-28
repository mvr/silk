import os
import torch
import numpy as np
import subprocess

this_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(this_dir, '../build')

class SilkNNUE(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = torch.nn.Embedding(7424, 128)
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
        assert tuple(x[0].shape) == (7424, 128)
        assert tuple(x[1].shape) == (32, 128)
        assert tuple(x[2].shape) == (32,)
        assert tuple(x[3].shape) == (32, 64)
        assert tuple(x[4].shape) == (32,)
        assert tuple(x[5].shape) == (1, 64)

        y0 = x[0]
        y1 = np.array([[x[1][i ^ (j >> 2), j] for j in range(128)] for i in range(32)], dtype=np.float32)
        y3 = np.array([[x[3][i ^ (j >> 2), (j >> 2) ^ ((j & 3) << 4)] for j in range(128)] for i in range(16)], dtype=np.float32)
        yb = np.array([[p for i in range(32) for p in [x[2][i], x[4][i], x[5][0, i], x[5][0, i + 32]]]], dtype=np.float32)

        return np.concatenate([y1, y3, yb, y0])

    def run_comparison(self, n_samples):

        x = ((torch.randn((n_samples, 32)) * 10000).to(torch.int32) & 255) + 256 * torch.arange(32).to(torch.int32)
        y_torch = self(x).cpu().detach().numpy().reshape(-1)

        nnue_filename = os.path.join(build_dir, 'test_nnue.dat')
        samples_filename = os.path.join(build_dir, 'test_samples.dat')
        program_filename = os.path.join(build_dir, 'src/testnet')
        output_filename = os.path.join(build_dir, 'test_outputs.dat')

        with open(nnue_filename, 'wb') as f:
            f.write(self.swizzle().tobytes())

        with open(samples_filename, 'wb') as f:
            f.write(x.cpu().detach().numpy().astype(np.uint32).tobytes())

        subprocess.check_call([program_filename, nnue_filename, samples_filename, output_filename, str(n_samples)])

        with open(output_filename, 'rb') as f:
            y_cuda = np.frombuffer(f.read(), np.float32)

        return y_cuda, y_torch


if __name__ == '__main__':

    nnue = SilkNNUE()
    y_cuda, y_torch = nnue.run_comparison(10000)
    rms_err = np.sqrt(np.square(y_cuda - y_torch).mean() / np.square(y_torch).mean())
    print('rms relative error: %.8f' % rms_err)
    assert(rms_err <= 1.0e-6)
