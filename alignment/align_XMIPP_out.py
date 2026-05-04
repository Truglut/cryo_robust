import torch
import argparse
import mrcfile
import pandas as pd
from io import StringIO
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def read_data(star_path, device="cuda"):
    """Calcula el promedio alineado de partículas con shift + rot en batches."""
    with open(star_path) as f:
        lines = f.readlines()

    optics_start = [i for i, l in enumerate(lines) if l.strip().startswith("data_optics")][0]
    particles_start = [i for i, l in enumerate(lines) if l.strip().startswith("data_particles")][0]

    # --- Optics ---
    optics_section = [
        l for l in lines[optics_start:particles_start]
        if l.strip() and not l.strip().startswith(("data_", "loop_", "#"))
    ]
    optics_cols = [l.strip() for l in optics_section if l.strip().startswith("_rln")]
    optics_data = [l for l in optics_section if not l.strip().startswith("_rln")]
    optics_str = " ".join(optics_cols) + "\n" + "".join(optics_data)
    optics_df = pd.read_csv(StringIO(optics_str), sep=r'\s+')
    pix_size = float(optics_df["_rlnImagePixelSize"].values[0])

    # --- Particles ---
    particles_section = [
        l for l in lines[particles_start:]
        if l.strip() and not l.strip().startswith(("data_", "loop_", "#"))
    ]
    particles_cols = [l.strip() for l in particles_section if l.strip().startswith("_rln")]
    particles_data = [l for l in particles_section if not l.strip().startswith("_rln")]
    particles_str = " ".join(particles_cols) + "\n" + "".join(particles_data)
    particles_df = pd.read_csv(StringIO(particles_str), sep=r'\s+')

    img_paths = particles_df["_rlnImageName"].tolist()
    psi = -particles_df["_rlnAnglePsi"].values
    shiftX = particles_df["_rlnOriginXAngst"].values / pix_size
    shiftY = particles_df["_rlnOriginYAngst"].values / pix_size

    # --- Cargar stack ---
    stack_path = img_paths[0].split("@")[1]
    with mrcfile.open(stack_path, permissive=True) as mrc:
        particles = mrc.data.copy()

    # Convertir tensores
    particles = torch.tensor(particles, dtype=torch.float32, device=device)
    angles_rad = torch.tensor(np.deg2rad(psi), dtype=torch.float32, device=device)
    shiftX = torch.tensor(shiftX, dtype=torch.float32, device=device)
    shiftY = torch.tensor(shiftY, dtype=torch.float32, device=device)

    return particles, angles_rad, shiftX, shiftY, pix_size


@torch.no_grad()
def fourier_shift_batch(imgs, shifts_x, shifts_y):

    n, h, w = imgs.shape
    device = imgs.device

    # Coordenadas de frecuencia
    ky = torch.fft.fftfreq(h, d=1.0, device=device).reshape(1, h, 1)
    kx = torch.fft.rfftfreq(w, d=1.0, device=device).reshape(1, 1, w//2 + 1)

    # Expandir shifts
    sx = shifts_x.view(n, 1, 1)
    sy = shifts_y.view(n, 1, 1)

    # Transformada real
    F = torch.fft.rfft2(imgs)  # (n,h,w//2+1), compleja

    # Fase para shift
    phase = torch.exp(-2j * torch.pi * (kx * sx + ky * sy))
    F.mul_(phase)  # inplace, ahorra memoria
    del phase

    # Transformada inversa real
    shifted = torch.fft.irfft2(F, s=(h, w))  # devuelve real
    del F

    return shifted


@torch.no_grad()
def align_particles(particles, angles_rad, shiftX, shiftY, device="cuda", batch_size=256):
    """Calcula el promedio alineado de partículas con shift + rot en batches."""
    
    n, h, w = particles.shape

    freq_w = w // 2 + 1
    
    # Tensor para guardar las imágenes reales (Shifted)
    # Usamos device='cpu' para no saturar la gráfica
    all_shifted = torch.empty((n, h, w), dtype=particles.dtype, device='cpu')

    # --- Base grid (solo una vez) ---
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # (h,w,2)
    base_grid_flat = base_grid.view(-1, 2)

    # --- Procesar por batches ---
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch = particles[i:j]
        batch_shx = shiftX[i:j]
        batch_shy = shiftY[i:j]
        batch_ang = angles_rad[i:j]

        # #rot
        cos = torch.cos(batch_ang)
        sin = torch.sin(batch_ang)
        rot_mats = torch.stack([
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin,  cos], dim=1)
        ], dim=1)  # (B, 2, 2)
        grids = base_grid_flat.unsqueeze(0).matmul(rot_mats.transpose(1, 2))
        grids = grids.view(-1, h, w, 2)

        # Rotar en batch
        imgs = batch.unsqueeze(1)  # (B, 1, H, W) 
        rotated = F.grid_sample(imgs, grids, align_corners=True, padding_mode="zeros")
        
        # --- 2) Shift después (en Fourier) ---
        rotated = rotated.squeeze(1)

        all_shifted[i:j] = fourier_shift_batch(rotated, batch_shx, batch_shy).detach().cpu()
                
        # Limpieza de memoria GPU
        del batch, grids, rotated, rot_mats
        torch.cuda.empty_cache()

    return all_shifted