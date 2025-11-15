import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from SegFormer import SegFormer_B0
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm


class TestSegmentationDataset(Dataset):
    def __init__(self, root_dir, subset='test'):
        self.image_dir = os.path.join(root_dir, "image", subset)
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*", "*.*"), recursive=True))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.to_tensor(img)
        return tensor, img_path


def load_model(weight_path, num_classes, device):
    model = SegFormer_B0(num_classes=num_classes)

    state_dict = torch.load(weight_path, map_location=device)

    model_keys = model.state_dict().keys()
    state_keys = state_dict.keys()

    # prefix mismatch
    has_module_prefix = any(k.startswith("module.") for k in state_keys)
    needs_prefix = any(k.startswith("module.") for k in model_keys)

    if has_module_prefix != needs_prefix:
        if needs_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[load_model] missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

    model = model.to(device)
    model.eval()
    return model


def save_prediction(pred, save_path, colormap_root, result_dir):
    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(pred_np).save(save_path)

    normed = pred_np.astype(np.float32) / 20
    cmap = cm.get_cmap('turbo')
    colored = cmap(normed)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)

    rel_path = os.path.relpath(save_path, start=os.path.join(result_dir, "label"))
    cmap_path = os.path.join(colormap_root, rel_path)
    os.makedirs(os.path.dirname(cmap_path), exist_ok=True)
    rgb_img.save(cmap_path)


def test(args):
    # ----- Device 선택 -----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    dataset = TestSegmentationDataset(args.dataset_dir, subset='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(args.weight_path, args.num_classes, device)

    colormap_root = os.path.join(args.result_dir, "colormap")

    for img_tensor, img_path in tqdm(dataloader, desc="Predicting..."):
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)

        # 결과 파일 경로 생성: dataset_dir/image/... → result_dir/image/...
        rel_path = os.path.relpath(img_path[0], os.path.join(args.dataset_dir, "image"))
        save_path = os.path.join(args.result_dir, "label", rel_path)

        save_prediction(pred, save_path, colormap_root, args.result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./SemanticDatasetTest",
                        help="Path to test images folder")
    parser.add_argument("--weight_path", type=str, default="./pths/0811_1/model_epoch500.pth",
                        help="Path to model weight (.pth)")
    parser.add_argument("--result_dir", type=str, default="./result2",
                        help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of segmentation classes")
    parser.add_argument("--input_size", type=int, nargs=2,
                        default=(1200, 1920), help="Input size (H W)")

    args = parser.parse_args()
    test(args)