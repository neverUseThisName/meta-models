import os

import torch


def test_forward():
    from meta_models.nn import ViT

    model = ViT(image_size=224, patch_size=16, num_classes=10, dim=256, depth=1, layer_depth=12, heads=4, mlp_dim=256)
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    assert y.shape == (1, 10)


def test_save_and_load():
    from meta_models.nn import ViT

    model = ViT(image_size=224, patch_size=16, num_classes=10, dim=256, depth=1, num_iters=12, heads=4, mlp_dim=256)
    torch.save(model.state_dict(), "meta_vit.pt")
    model.load_state_dict(torch.load("meta_vit.pt"))
    os.remove("meta_vit.pt")
