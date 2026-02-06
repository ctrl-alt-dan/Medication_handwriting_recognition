import os
import torch
import torch.nn as nn


class TuneCNN(nn.Module):
    """
    Matches your DL-3 TuneCNN architecture:
    3 conv blocks (each: Conv-BN-ReLU x2 + MaxPool + Dropout),
    then AdaptiveAvgPool -> FC head.
    """
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

        self.features = nn.Sequential(
            block(1, 32, pool=(2, 2)),
            block(32, 64, pool=(2, 2)),
            block(64, 128, pool=(2, 2)),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_model(weights_path: str, num_classes: int | None = None):
    """
    Loads model weights. If num_classes isn't provided, we infer from the checkpoint
    when possible (otherwise default to 78).
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Place best_tuned_cnn.pt next to app.py (or update the path in the sidebar)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to infer num_classes from checkpoint (best effort)
    ckpt = torch.load(weights_path, map_location="cpu")
    inferred = None
    if isinstance(ckpt, dict):
        # common case: state_dict
        sd = ckpt
        # try locate final layer weight
        for key in ["classifier.4.weight", "fc.weight", "head.weight"]:
            if key in sd:
                inferred = sd[key].shape[0]
                break

    ncls = int(num_classes or inferred or 78)

    model = TuneCNN(num_classes=ncls, dropout=0.3).to(device)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt, strict=True)
    model.eval()
    return model, device


@torch.inference_mode()
def predict_topk(model: nn.Module, x: torch.Tensor, top_k: int = 5, device=None):
    """
    x: [1,1,64,384] float tensor
    returns list[(class_idx:int, prob:float)] sorted desc
    """
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    k = min(int(top_k), probs.numel())
    p, idx = torch.topk(probs, k=k)
    out = [(int(i.item()), float(pp.item())) for i, pp in zip(idx, p)]
    return out
