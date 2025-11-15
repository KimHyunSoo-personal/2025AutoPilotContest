import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from AFFormer_full_ver2 import PureAFFormerSmall
from functions import *
from pathlib import Path


def safe_load_partial(model, state_dict, verbose=True, drop_prefixes=None):
    import re
    if drop_prefixes is None:
        drop_prefixes = []

    sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        sd[k] = v

    remap = {}
    for k in list(sd.keys()):
        nk = k
        remap[k] = nk

    sd2 = {remap[k]: v for k, v in sd.items()}

    msd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in sd2.items():
        if any(k.startswith(p) for p in drop_prefixes):
            skipped.append(k)
            continue
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if verbose:
        print(f"[safe_load] loaded={len(filtered)}, skipped={len(skipped)}, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if skipped[:10]:
            print("  skipped(sample):", skipped[:10])
        if missing[:10]:
            print("  missing(sample):", missing[:10])
        if unexpected[:10]:
            print("  unexpected(sample):", unexpected[:10])
    return missing, unexpected


def train(args):
    # ----- Device 선택 -----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # Dataset & Dataloader
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
    display_dataset_info(args.dataset_dir, train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0, pin_memory=True)

    # Model
    model = PureAFFormerSmall(num_classes=args.num_classes).to(device)

    if args.loadpath is not None:
        ckpt = torch.load(args.loadpath, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        safe_load_partial(model, state_dict, verbose=True, drop_prefixes=None)

    # Loss, Optimizer, Scheduler
    criterion = CrossEntropy(ignore_label=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")

    min_loss = 100.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"[{device}] Epoch [{epoch+1}/{args.epochs}]", ncols=100)

        for i, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)

            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if isinstance(outputs, (tuple, list)):
                main, aux = outputs
                loss = criterion(main, labels) + 0.4 * criterion(aux, labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(),
                             avg_loss=total_loss/(i+1),
                             lr=scheduler.get_last_lr()[0])

        torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < min_loss:
            min_loss = avg_loss
            ckp_path = os.path.join(args.result_dir, f"model_best.pth")
            torch.save(model.state_dict(), ckp_path)

        if (epoch + 1) % 20 == 0:
            ckp_path = os.path.join(args.result_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckp_path)
            with open(log_path, "a") as f:
                f.write("\n%d\t\t%.4f\t\t%.8f" %
                        (epoch + 1, avg_loss, scheduler.get_last_lr()[0]))


# ---------- Argparse ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,
                        default="/Users/hyunsookim/dataset/SemanticDataset_final_sampled")
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default="./result_afformer")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[512, 1024],
                        type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.5],
                        type=arg_as_list, help="resize Input")

    args = parser.parse_args()

    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    train(args)