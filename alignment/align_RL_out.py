import argparse
import torch
import torch.nn.functional as F
import mrcfile
import numpy as np
import pandas as pd
from io import StringIO


def fourier_shift(img, shift_x, shift_y):
    """
    Aplica un shift (en píxeles) a una imagen 2D usando traslación en Fourier.
    """
    h, w = img.shape
    ky = torch.fft.fftfreq(h, d=1.0, device=img.device).reshape(-1, 1)
    kx = torch.fft.fftfreq(w, d=1.0, device=img.device).reshape(1, -1)

    phase = torch.exp(-2j * torch.pi * (kx * shift_x + ky * shift_y))
    F_img = torch.fft.fft2(img)
    F_shifted = F_img * phase
    return torch.fft.ifft2(F_shifted).real



def read_data(star_path, device="cpu"):
    with open(star_path) as f:
        lines = f.readlines()

    # separar secciones optics y particles
    optics_start = [
        i for i, l in enumerate(lines) if l.strip().startswith("data_optics")
    ][0]
    particles_start = [
        i for i, l in enumerate(lines) if l.strip().startswith("data_particles")
    ][0]

    # --- Extraer bloque optics ---
    optics_section = [
        l
        for l in lines[optics_start:particles_start]
        if l.strip() and not l.strip().startswith(("data_", "loop_", "#"))
    ]
    optics_cols = [l.strip() for l in optics_section if l.strip().startswith("_rln")]
    optics_data = [l for l in optics_section if not l.strip().startswith("_rln")]
    optics_str = " ".join(optics_cols) + "\n" + "".join(optics_data)
    optics_df = pd.read_csv(StringIO(optics_str), sep=r"\s+")

    # pixel size
    pix_size = float(optics_df["_rlnImagePixelSize"].values[0])

    # --- Extraer bloque particles ---
    particles_section = [
        l
        for l in lines[particles_start:]
        if l.strip() and not l.strip().startswith(("data_", "loop_", "#"))
    ]
    particles_cols = [
        l.strip() for l in particles_section if l.strip().startswith("_rln")
    ]
    particles_data = [l for l in particles_section if not l.strip().startswith("_rln")]
    particles_str = " ".join(particles_cols) + "\n" + "".join(particles_data)
    particles_df = pd.read_csv(StringIO(particles_str), sep=r"\s+")

    # --- Extraer campos ---
    img_paths = particles_df["_rlnImageName"].tolist()
    psi = -particles_df["_rlnAnglePsi"].values
    shiftX = particles_df["_rlnOriginXAngst"].values / pix_size
    shiftY = particles_df["_rlnOriginYAngst"].values / pix_size

    # --- Cargar stack de partículas ---
    stack_path = img_paths[0].split("@")[1]  # formato: "XXX@file.mrcs"
    with mrcfile.open(stack_path, permissive=True) as mrc:
        particles = mrc.data.copy()

    particles = torch.tensor(particles, dtype=torch.float32, device=device)

    psi = torch.tensor(np.deg2rad(psi), dtype=torch.float32, device=device)
    shiftX = torch.tensor(shiftX, dtype=torch.float32, device=device)
    shiftY = torch.tensor(shiftY, dtype=torch.float32, device=device)

    return particles, psi, shiftX, shiftY


def align_particles(star_path, device="cpu"):
    particles, psi, shiftX, shiftY = read_data(star_path, device=device)

    n, h, w = particles.shape

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # (h,w,2)

    aligned = []
    for i in range(n):
        shifted = fourier_shift(particles[i], shiftX[i], shiftY[i])

        # 2. Rotación
        cos, sin = torch.cos(psi[i]), torch.sin(psi[i])
        rot_mat = torch.tensor([[cos, -sin], [sin, cos]], device=device)
        grid = base_grid @ rot_mat.T
        grid = grid.unsqueeze(0)

        img = shifted.unsqueeze(0).unsqueeze(0)  # agregar dims batch+channel
        rotated = F.grid_sample(img, grid, align_corners=True, padding_mode="zeros")[
            0, 0
        ]

        aligned.append(rotated)

    aligned = torch.stack(aligned, dim=0)

    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="Promediar partículas alineadas de un archivo .star de RELION 3.0+"
    )
    parser.add_argument("star", type=str, help="Ruta al archivo .star de entrada")
    parser.add_argument(
        "--out",
        type=str,
        default="aligned_particles.mrcs",
        help="Ruta de salida para el .mrc promedio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo para PyTorch",
    )
    args = parser.parse_args()

    aligned = align_particles(args.star, device=args.device)

    mrcfile.write(args.out, data=aligned.detach().cpu().numpy())
    print(f"Partículas alineadas guardadas en {args.out}")

if __name__ == "__main__":
    main()
