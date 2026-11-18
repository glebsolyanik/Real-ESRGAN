import argparse
import math
import random
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

PATCH_NAME = re.compile(r"^(\d+)_Crop_(\d+)_y(\d+)_x(\d+)")
LR_SCALE = 4
DEFAULT_OUT = "datasets/BC-MultiSet-50_SR"
SHEET_COLUMNS = 8
SHEET_TILE = 192
SHEET_MAX_IMAGES = 48


def main():
    args = parse_args()
    degrade = TrainingDegradation(args.degradation_opt) if args.degradation_opt else None
    scale = degrade.opt["scale"] if degrade else LR_SCALE
    default_out = DEFAULT_OUT if scale == LR_SCALE else f"{DEFAULT_OUT}_x{scale}"
    out_dir = Path(default_out if args.out is None else args.out)
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    patches = measure_all_patches(Path(args.root))
    patches.to_csv(report_dir / "stats.csv", index=False)

    print("\n=== ПРОВЕРКИ ===")
    check_unreadable(patches)
    check_slide_leakage(patches)
    check_slides_are_patients(patches, Path(args.clinical), args.patient_col)

    patches, blur_threshold = mark_excluded(patches, args.bg_max, args.blur_pct)
    patches[patches["reason"] != ""].to_csv(report_dir / "excluded.csv", index=False)
    print_filtering_summary(patches, args.bg_max, args.blur_pct, blur_threshold)
    save_threshold_sheets(patches, report_dir)

    kept = patches[patches["reason"] == ""]
    train = write_train_meta_info(kept, args.train_folds.split(","), out_dir)
    val = write_val_pairs(kept, args.val_fold, args.n_val, args.seed, out_dir, degrade)
    val[["name", "fold", "slide"]].to_csv(report_dir / "val_list.csv", index=False)

    print("\n=== ГОТОВО ===")
    print(f"train: {len(train)} картинок -> {out_dir / 'meta_info' / 'train.txt'}")
    print(f"val:   {len(val)} пар HR/LR  -> {out_dir / 'val_hr'}, {out_dir / 'val_lr'}")
    if args.degradation_opt:
        print(f"       + деградированные LR -> {out_dir / 'val_lr_degraded'}")
    print(f"отчёт: {report_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/BC-MultiSet-50")
    parser.add_argument("--out", help="Каталог результата. По умолчанию зависит от масштаба в конфиге.")
    parser.add_argument("--train_folds", default="fold1,fold2,fold3")
    parser.add_argument("--val_fold", default="fold4")
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--bg_max", type=float, default=0.8)
    parser.add_argument("--blur_pct", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clinical", default="datasets/BC-MultiSet-50/BC-MultiSet.xlsx")
    parser.add_argument("--patient_col", default="ID", help="Колонка в --clinical с ID пациента")
    parser.add_argument(
        "--degradation_opt",
        default="options/finetune_realesrgan_x4plus_bc_multiset.yml",
        help="Конфиг обучения, из которого берутся параметры деградаций для val_lr_degraded ('' — не делать)")
    return parser.parse_args()


def measure_all_patches(root):
    fold_dirs = sorted(path for path in root.glob("fold*") if path.is_dir())
    if not fold_dirs:
        raise SystemExit(f"Не нашёл папки fold* в {root.resolve()}. Проверь --root.")

    rows = []
    for fold_dir in fold_dirs:
        image_paths = find_fold_images(fold_dir)
        print(f"{fold_dir.name}: {len(image_paths)} картинок, считаю метрики...")
        for path in image_paths:
            rows.append({
                "path": str(path.resolve()),
                "name": path.name,
                "fold": fold_dir.name,
                "slide": slide_id(path.name),
                **measure_patch(path),
            })
    return pd.DataFrame(rows)


def find_fold_images(fold_dir):
    images_dir = fold_dir / "images"
    return sorted((images_dir if images_dir.is_dir() else fold_dir).glob("*.png"))


def slide_id(file_name):
    match = PATCH_NAME.match(file_name)
    return match.group(1) if match else None


def measure_patch(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "bg_frac": np.nan, "sharp": np.nan}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = background_mask(image)
    return {
        "ok": True,
        "bg_frac": float(background.mean()),
        "sharp": tissue_sharpness(gray, background),
    }


def background_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    is_colorless = hsv[..., 1] < 25
    is_bright = hsv[..., 2] > 200
    return is_colorless & is_bright


def tissue_sharpness(gray, background):
    tissue = (~background).astype(np.uint8)
    tissue_away_from_edges = cv2.erode(tissue, np.ones((7, 7), np.uint8)).astype(bool)
    if tissue_away_from_edges.mean() < 0.05:
        return np.nan
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[tissue_away_from_edges].var())


def check_unreadable(patches):
    print(f"Нечитаемых картинок: {(~patches['ok']).sum()}")


def check_slide_leakage(patches):
    if patches["slide"].isna().any():
        print("Имена файлов не в формате <slide>_Crop_..., проверку утечки пропускаю.")
        return

    folds_per_slide = patches.groupby("slide")["fold"].nunique()
    leaked_slides = list(folds_per_slide[folds_per_slide > 1].index)
    slides_per_fold = patches.groupby("fold")["slide"].nunique().to_dict()
    print(f"Слайдов всего: {len(folds_per_slide)}; по фолдам: {slides_per_fold}")

    if leaked_slides:
        print(f"!!! ВНИМАНИЕ: {len(leaked_slides)} слайдов в нескольких фолдах, val будет нечестным: {leaked_slides[:10]}")
    else:
        print("OK: каждый слайд ровно в одном фолде, утечки между train и val нет.")


def check_slides_are_patients(patches, clinical_path, patient_col):
    # Проверка утечки по слайдам равна проверке по пациентам, только если номер слайда
    # в имени патча — это ID пациента из клинической таблицы (один слайд на пациента).
    if not clinical_path.is_file():
        print(f"Нет таблицы {clinical_path}, сверку слайдов с пациентами пропускаю.")
        return
    if patches["slide"].isna().any():
        print("Номера слайдов не извлечены из имён файлов, сверку с пациентами пропускаю.")
        return

    table = pd.read_excel(clinical_path)
    if patient_col not in table.columns:
        print(f"Нет колонки '{patient_col}' в {clinical_path.name}, укажи --patient_col. Колонки: {list(table.columns)}")
        return

    patient_ids = set(table[patient_col].map(normalize_id).dropna())
    slide_ids = set(patches["slide"])
    slides_not_in_table = sorted(slide_ids - patient_ids)
    patients_without_patches = sorted(patient_ids - slide_ids)
    print(f"Пациентов в таблице: {len(patient_ids)}, слайдов в патчах: {len(slide_ids)}")

    if slides_not_in_table:
        print(f"!!! {len(slides_not_in_table)} слайдов не совпадают с ID пациентов: {slides_not_in_table[:10]}. "
              "Номер в имени патча может быть не ID пациента, проверка по слайдам может пропустить утечку.")
    else:
        print("OK: номер слайда в имени патча = ID пациента, проверка по слайдам — это проверка по пациентам.")
    if patients_without_patches:
        print(f"Пациенты из таблицы без патчей: {patients_without_patches[:10]}")


def normalize_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def mark_excluded(patches, bg_max, blur_pct):
    patches = patches.copy()
    patches["reason"] = ""
    still_kept = lambda: patches["reason"] == ""

    patches.loc[~patches["ok"], "reason"] = "unreadable"
    patches.loc[still_kept() & (patches["bg_frac"] >= bg_max), "reason"] = "background"
    patches.loc[still_kept() & patches["sharp"].isna(), "reason"] = "no_tissue"

    blur_threshold = float(np.nanpercentile(patches.loc[still_kept(), "sharp"], blur_pct))
    patches.loc[still_kept() & (patches["sharp"] < blur_threshold), "reason"] = "blur"
    return patches, blur_threshold


def print_filtering_summary(patches, bg_max, blur_pct, blur_threshold):
    print("\n=== ФИЛЬТРАЦИЯ ===")
    print(f"Порог фона: >= {bg_max:.0%}; порог резкости: < {blur_threshold:.1f} (нижние {blur_pct}%)")
    print(patches["reason"].replace("", "kept").value_counts().to_string())


def save_threshold_sheets(patches, report_dir):
    excluded_background = patches[patches["reason"] == "background"].sort_values("bg_frac")
    excluded_blur = patches[patches["reason"] == "blur"].sort_values("sharp", ascending=False)
    kept_least_sharp = patches[patches["reason"] == ""].sort_values("sharp")

    save_contact_sheet(excluded_background, report_dir / "1_excluded_background.png")
    save_contact_sheet(excluded_blur, report_dir / "2_excluded_blur.png")
    save_contact_sheet(kept_least_sharp, report_dir / "3_kept_least_sharp.png")


def save_contact_sheet(patches, out_path):
    patches = patches.head(SHEET_MAX_IMAGES)
    if patches.empty:
        return

    cell_height = SHEET_TILE + 18
    rows = (len(patches) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = np.full((rows * cell_height, SHEET_COLUMNS * SHEET_TILE, 3), 255, np.uint8)

    for index, (_, patch) in enumerate(patches.iterrows()):
        top = (index // SHEET_COLUMNS) * cell_height
        left = (index % SHEET_COLUMNS) * SHEET_TILE
        tile = cv2.resize(cv2.imread(patch["path"]), (SHEET_TILE, SHEET_TILE), interpolation=cv2.INTER_AREA)
        sheet[top:top + SHEET_TILE, left:left + SHEET_TILE] = tile
        label = f"s={patch['sharp']:.0f} bg={patch['bg_frac']:.2f}"
        cv2.putText(sheet, label, (left + 3, top + SHEET_TILE + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    cv2.imwrite(str(out_path), sheet)


def write_train_meta_info(kept, train_folds, out_dir):
    train = kept[kept["fold"].isin(train_folds)]
    meta_dir = out_dir / "meta_info"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "train.txt").write_text("\n".join(train["path"]))
    return train


def write_val_pairs(kept, val_fold, n_val, seed, out_dir, degrade):
    candidates = kept[kept["fold"] == val_fold]
    val = candidates.sample(n=min(n_val, len(candidates)), random_state=seed)
    total = len(val)
    names = set(val["name"])
    if len(names) != total:
        duplicates = val.loc[val["name"].duplicated(), "name"].tolist()
        raise ValueError(f"Повторяющиеся имена файлов в {val_fold}: {duplicates}")

    scale = degrade.opt["scale"] if degrade else LR_SCALE
    folders = ["val_hr", "val_lr"]
    if degrade:
        folders.append("val_lr_degraded")

    for folder in folders:
        target = out_dir / folder
        if target.is_dir():
            extra = sorted(p.name for p in target.iterdir()
                           if p.is_file() and not p.name.startswith(".") and p.suffix != ".png")
            if extra:
                raise ValueError(f"Посторонние файлы в {target}: {extra}")
        elif target.exists():
            raise NotADirectoryError(target)

    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".val-", dir=out_dir) as staging:
        staged = Path(staging)
        for folder in folders:
            (staged / folder).mkdir()

        print(f"Валидация {scale}x: 0/{total} патчей", flush=True)
        for index, (_, patch) in enumerate(val.iterrows()):
            try:
                hr = cv2.imread(patch["path"])
                if hr is None:
                    raise OSError("OpenCV не смог прочитать HR")
                images = [hr, downscale(hr, scale)]
                if degrade:
                    images.append(degrade(hr, seed + index))
                for folder, image in zip(folders, images):
                    path = staged / folder / patch["name"]
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f"OpenCV не смог записать {folder}/{path.name}")
            except Exception as exc:
                raise RuntimeError(f"Ошибка подготовки валидации для {patch['path']}: {exc}") from exc

            done = index + 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"Валидация {scale}x: {done}/{total} патчей", flush=True)

        # Keep the previous validation set if image generation fails.
        for folder in folders:
            target = out_dir / folder
            target.mkdir(exist_ok=True)
            for path in (staged / folder).iterdir():
                path.replace(target / path.name)
            for path in target.glob("*.png"):
                if path.name not in names:
                    path.unlink()
    return val


def downscale(image, scale=LR_SCALE):
    height, width = image.shape[:2]
    return cv2.resize(image, (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)


class TrainingDegradation:
    """Двухстадийная деградация Real-ESRGAN, как на обучении, но с фиксированным seed на картинку.

    Повторяет генерацию ядер из RealESRGANDataset.__getitem__ и синтез LR из RealESRGANModel.feed_data,
    параметры берёт из того же yml, что и обучение.
    """

    KERNEL_RANGE = [2 * v + 1 for v in range(3, 11)]  # 7..21, как в RealESRGANDataset

    def __init__(self, opt_path):
        import torch
        import yaml
        from basicsr.utils import DiffJPEG, USMSharp

        # Small patches processed one at a time do not benefit from a large CPU thread pool.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        config = yaml.safe_load(Path(opt_path).read_text())
        self.opt = {**config, **config["datasets"]["train"]}
        self.torch = torch
        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpener = USMSharp()
        print(f"val_lr_degraded: параметры из {opt_path}, потоков PyTorch: {torch.get_num_threads()}", flush=True)

    def __call__(self, hr_bgr, seed):
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        gt = torch.from_numpy(hr_bgr[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.
        with torch.no_grad():
            lq = self.synthesize_lq(gt)
        lq = (lq[0].numpy().transpose(1, 2, 0)[..., ::-1] * 255.).round().astype(np.uint8)
        return lq

    def random_kernel(self, suffix):
        from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

        opt = self.opt
        kernel_size = random.choice(self.KERNEL_RANGE)
        if np.random.uniform() < opt[f"sinc_prob{suffix}"]:
            omega_c = np.random.uniform(np.pi / 3 if kernel_size < 13 else np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                opt[f"kernel_list{suffix}"], opt[f"kernel_prob{suffix}"], kernel_size,
                opt[f"blur_sigma{suffix}"], opt[f"blur_sigma{suffix}"], [-math.pi, math.pi],
                opt[f"betag_range{suffix}"], opt[f"betap_range{suffix}"], noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return self.torch.FloatTensor(kernel).unsqueeze(0)

    def random_sinc_kernel(self):
        from basicsr.data.degradations import circular_lowpass_kernel

        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.KERNEL_RANGE)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            return self.torch.FloatTensor(circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)).unsqueeze(0)
        pulse = self.torch.zeros(1, 21, 21)
        pulse[0, 10, 10] = 1
        return pulse

    def synthesize_lq(self, gt):
        import torch
        from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
        from basicsr.utils.img_process_util import filter2D
        from torch.nn import functional as F

        opt = self.opt
        kernel1, kernel2, sinc_kernel = self.random_kernel(""), self.random_kernel("2"), self.random_sinc_kernel()
        ori_h, ori_w = gt.shape[2:4]

        def random_scale(resize_prob, resize_range):
            updown_type = random.choices(["up", "down", "keep"], resize_prob)[0]
            if updown_type == "up":
                return np.random.uniform(1, resize_range[1])
            if updown_type == "down":
                return np.random.uniform(resize_range[0], 1)
            return 1

        def add_noise(out, suffix):
            if np.random.uniform() < opt[f"gaussian_noise_prob{suffix}"]:
                return random_add_gaussian_noise_pt(
                    out, sigma_range=opt[f"noise_range{suffix}"], clip=True, rounds=False,
                    gray_prob=opt[f"gray_noise_prob{suffix}"])
            return random_add_poisson_noise_pt(
                out, scale_range=opt[f"poisson_scale_range{suffix}"], gray_prob=opt[f"gray_noise_prob{suffix}"],
                clip=True, rounds=False)

        def jpeg(out, jpeg_range):
            quality = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            return self.jpeger(torch.clamp(out, 0, 1), quality=quality)

        def resize_back_and_sinc(out):
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(ori_h // opt["scale"], ori_w // opt["scale"]), mode=mode)
            return filter2D(out, sinc_kernel)

        # первая деградация (на обучении деградирует GT после USM, повторяем это)
        out = filter2D(self.usm_sharpener(gt), kernel1)
        scale = random_scale(opt["resize_prob"], opt["resize_range"])
        out = F.interpolate(out, scale_factor=scale, mode=random.choice(["area", "bilinear", "bicubic"]))
        out = add_noise(out, "")
        out = jpeg(out, opt["jpeg_range"])

        # вторая деградация
        if np.random.uniform() < opt["second_blur_prob"]:
            out = filter2D(out, kernel2)
        scale = random_scale(opt["resize_prob2"], opt["resize_range2"])
        size = (int(ori_h / opt["scale"] * scale), int(ori_w / opt["scale"] * scale))
        out = F.interpolate(out, size=size, mode=random.choice(["area", "bilinear", "bicubic"]))
        out = add_noise(out, "2")

        # [resize back + sinc] и JPEG в случайном порядке
        if np.random.uniform() < 0.5:
            out = jpeg(resize_back_and_sinc(out), opt["jpeg_range2"])
        else:
            out = resize_back_and_sinc(jpeg(out, opt["jpeg_range2"]))

        return torch.clamp((out * 255.0).round(), 0, 255) / 255.


if __name__ == "__main__":
    main()
