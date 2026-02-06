# model.py
import torch
import torch.nn as nn


class TuneCNN(nn.Module):
    def __init__(self, num_classes=78, dropout=0.3):
        super().__init__()

        def block(cin, cout, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
                nn.Dropout(dropout),
            )

        # EXACT match to your notebook
        self.features = nn.Sequential(
            block(1, 32, pool=(2, 2)),
            block(32, 64, pool=(2, 2)),
            block(64, 128, pool=(2, 2)),
            block(128, 256, pool=(2, 1)),  # preserve width
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # IMPORTANT: called "head" (matches weights)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def load_model(weights_path: str, num_classes: int = 78, device: str = "cpu") -> TuneCNN:
    """
    Loads the trained TuneCNN model with weights from best_tuned_cnn.pt
    Handles common checkpoint formats.
    """
    model = TuneCNN(num_classes=num_classes)
    model.to(device)

    state = torch.load(weights_path, map_location=device)

    # Handle common save formats
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def predict_topk(model: nn.Module, x: torch.Tensor, k: int = 5):
    """
    x: torch.Tensor of shape [1, 1, 64, 384] (already preprocessed)
    Returns:
      topk_idx: np.ndarray shape [k]
      topk_probs: np.ndarray shape [k] (0..1)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B,C,H,W], got {tuple(x.shape)}")

    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    topk_probs, topk_idx = torch.topk(probs, k=k, dim=1)
    return topk_idx[0].cpu().numpy(), topk_probs[0].cpu().numpy()
