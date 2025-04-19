import torch
import prev

torch.save(prev.model.state_dict(), "model.pth")
