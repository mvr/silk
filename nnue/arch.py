import os
import torch
import numpy as np
import subprocess
from math import gcd

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

    def train_loop(self, mb_size=4096, init_lr=0.002, n_epochs=2):

        self.cuda()

        dataset_filename = os.path.join(build_dir, '../dataset.bin')
        with open(dataset_filename, 'rb') as f:
            dataset = np.frombuffer(f.read(), np.uint8)
        dataset = dataset.reshape(-1, 32)
        print(dataset.shape)

        l = int(dataset.shape[0])
        s = int(l * (1.25 ** 0.5 - 0.5)) | 1
        while gcd(l, s) > 1:
            s += 2

        assert(l % mb_size == 0)

        print("total samples = %d, stride = %d" % (l, s))

        # do two complete epochs over the data:
        n_batches = (n_epochs * l) // mb_size

        optimizer = torch.optim.Adam(self.parameters(), lr=init_lr, weight_decay=1.0e-5, eps=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches, eta_min=0)

        stuff = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0, 0, 0],
            [3, 2, 1, 0, 7, 6, 5, 4, 16, 19, 18, 17, 12, 15, 14, 13,  8, 11, 10,  9, 20, 23, 22, 21, 27, 26, 25, 24, 28, 0, 0, 0],
            [1, 2, 3, 0, 5, 6, 7, 4,  9, 10, 11,  8, 13, 14, 15, 12, 17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 28, 0, 0, 0],
            [0, 3, 2, 1, 4, 7, 6, 5, 17, 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10, 21, 20, 23, 22, 24, 27, 26, 25, 28, 0, 0, 0],
            [2, 3, 0, 1, 6, 7, 4, 5, 10, 11,  8,  9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 28, 0, 0, 0],
            [1, 0, 3, 2, 5, 4, 7, 6, 18, 17, 16, 19, 14, 13, 12, 15, 10,  9,  8, 11, 22, 21, 20, 23, 25, 24, 27, 26, 28, 0, 0, 0],
            [3, 0, 1, 2, 7, 4, 5, 6, 11,  8,  9, 10, 15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22, 27, 24, 25, 26, 28, 0, 0, 0],
            [2, 1, 0, 3, 6, 5, 4, 7, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8, 23, 22, 21, 20, 26, 25, 24, 27, 28, 0, 0, 0]
        ], dtype=np.uint32)[:, None, :] * 256

        coeffs = np.array([[0.5 ** 24], [0.5 ** 16], [0.5 ** 8]], dtype=np.float32)
        coeffs = torch.from_numpy(coeffs).cuda()

        stuff = torch.from_numpy(stuff.astype(np.int32)).cuda()

        idxs = np.array([(i * s) % l for i in range(mb_size)], dtype=np.int64)
        s2 = (mb_size * s) % l

        for i in range(n_batches):

            optimizer.zero_grad()

            batch = dataset[idxs].astype(np.int32) # upcast from uint8 to int32
            batch = torch.from_numpy(batch).cuda()

            x = (batch + stuff).reshape(-1, 32)
            y = torch.matmul(x[:, -3:].to(torch.float32), coeffs)
            y_pred = self(x)

            idxs = (idxs + s2) % l

            loss = torch.square(y_pred - y).mean()
            denom = torch.square(y - y.mean()).mean()

            loss.backward()
            print(('R^2 = %.2f' % (100.0 * (1.0 - (loss / denom).item()))) + '%')

            optimizer.step()
            scheduler.step()

            # current_lr = scheduler.get_last_lr()[0]
            # print('Current LR: %.8f' % current_lr)

        self.cpu()


if __name__ == '__main__':

    nnue = SilkNNUE()
    nnue.embedding.weight.data.mul_(1.0e-3) # override bad initialisation

    nnue.train_loop()

    y_cuda, y_torch = nnue.run_comparison(10000)
    rms_err = np.sqrt(np.square(y_cuda - y_torch).mean() / np.square(y_torch).mean())
    print('rms relative error: %.8f' % rms_err)
    assert(rms_err <= 1.0e-6)
