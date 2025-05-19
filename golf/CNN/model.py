import torch.nn as nn

class CNN_GRU_Classifier(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super().__init__()
        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.flattened_size = 32 * 56 * 56  # assuming input is 224x224
        self.gru = nn.GRU(input_size=self.flattened_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)  # reshape for GRU
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 마지막 타임스텝 출력
        out = self.fc(out)
        return out
