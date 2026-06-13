"""
Alignment code for particles whose alignment parameters have been calculated by RELION

Original author: Erney Ramírez Aportela
The code has been modified by Andrés Contreras Santos to split data reading and 
alignment into two different functions.
"""

import argparse
import torch
import torch.nn.functional as F
import mrcfile
import numpy as np
import pandas as pd
import os
from io import StringIO


def fourier_shift_batch(
    imgs: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor
) -> torch.Tensor:
    """
    Apply subpixel shifts to a batch of 2D images using the Fourier shift theorem.

    Parameters
    ----------
    imgs : torch.Tensor
        Input tensor of shape (N, H, W) containing a batch of 2D images.
    shift_x : torch.Tensor
        Tensor of shape (N,) with translations along the X axis (columns),
        expressed in pixels.
    shift_y : torch.Tensor
        Tensor of shape (N,) with translations along the Y axis (rows),
        expressed in pixels.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, H, W) containing the shifted images.

    Notes
    -----
    The shifts are applied in the Fourier domain by multiplying the real FFT
    of each image by a complex phase factor. This enables subpixel-accurate
    translations without interpolation artifacts.

    A real-to-complex FFT (`rfft2`) is used for efficiency, exploiting the
    Hermitian symmetry of real-valued inputs. The shifted images are recovered
    via the inverse real FFT (`irfft2`).
    """
    n, h, w = imgs.shape

    # Frequency coordinates
    ky = torch.fft.fftfreq(h, d=1.0, device=imgs.device).reshape(1, h, 1)
    kx = torch.fft.rfftfreq(w, d=1.0, device=imgs.device).reshape(1, 1, w // 2 + 1)

    # Expand shifts
    sx = shift_x.view(n, 1, 1)
    sy = shift_y.view(n, 1, 1)

    # Calculate phase and shift images
    phase = torch.exp(-2j * torch.pi * (kx * sx + ky * sy))
    fourier_images = torch.fft.rfft2(imgs)
    fourier_images.mul_(phase)
    del phase

    # Return real space images
    return torch.fft.irfft2(fourier_images, s=(h, w))


def read_data(
    star_path: str, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read a .star file and return the particle stack along with alignment
    parameters (in-plane rotation and translations).

    Parameters
    ----------
    star_path : str
        Path to the .star file.
    device : str, optional
        PyTorch device where tensors will be loaded (e.g., "cpu" or "cuda"),
        by default "cpu".

    Returns
    -------
    particles : torch.Tensor
        Tensor of shape (N, H, W) containing the particle images.
    psi : torch.Tensor
        Tensor of shape (N,) with in-plane rotation angles in radians.
    shiftX : torch.Tensor
        Tensor of shape (N,) with translations along the X axis in pixels.
    shiftY : torch.Tensor
        Tensor of shape (N,) with translations along the Y axis in pixels.
    """
    with open(star_path) as f:
        lines = f.readlines()

    # Separate optics and particles sections
    optics_start = [
        i for i, l in enumerate(lines) if l.strip().startswith("data_optics")
    ][0]
    particles_start = [
        i for i, l in enumerate(lines) if l.strip().startswith("data_particles")
    ][0]

    # Extract optics block
    optics_section = [
        l
        for l in lines[optics_start:particles_start]
        if l.strip() and not l.strip().startswith(("data_", "loop_", "#"))
    ]
    optics_cols = [l.strip() for l in optics_section if l.strip().startswith("_rln")]
    optics_data = [l for l in optics_section if not l.strip().startswith("_rln")]
    optics_str = " ".join(optics_cols) + "\n" + "".join(optics_data)
    optics_df = pd.read_csv(StringIO(optics_str), sep=r"\s+")

    # Pixel size
    pix_size = float(optics_df["_rlnImagePixelSize"].values[0])

    # Extract particles block
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

    # Extract fields
    img_paths = particles_df["_rlnImageName"].tolist()
    psi = -particles_df["_rlnAnglePsi"].values
    shiftX = particles_df["_rlnOriginXAngst"].values / pix_size
    shiftY = particles_df["_rlnOriginYAngst"].values / pix_size

    # Load particles stack
    star_dir = os.path.dirname(star_path)
    pure_stack_path = img_paths[0].split("@")[1]
    stack_path = os.path.join(star_dir, pure_stack_path)
    with mrcfile.open(stack_path, permissive=True) as mrc:
        particles = mrc.data.copy()

    # Convert everything to tensors
    particles = torch.tensor(particles, dtype=torch.float32, device=device)
    psi = torch.tensor(np.deg2rad(psi), dtype=torch.float32, device=device)
    shiftX = torch.tensor(shiftX, dtype=torch.float32, device=device)
    shiftY = torch.tensor(shiftY, dtype=torch.float32, device=device)

    return particles, psi, shiftX, shiftY, pix_size


def align_particles_batch_RELION(
    particles: torch.Tensor,
    psi: torch.Tensor,
    shiftX: torch.Tensor,
    shiftY: torch.Tensor,
    batch_size: int = 256,
    inplace: bool = True,
):
    """
    Aligns a set of particles using batched Fourier shifts and spatial rotations.
    Follows RELION's conventions for alignment.

    The alignment consists of:
    1. Subpixel translations applied in Fourier space.
    2. In-plane rotations applied via grid sampling.

    Parameters
    ----------
    particles : torch.Tensor
        Tensor of shape (N, H, W) containing the unaligned particle images.
    psi : torch.Tensor
        Tensor of shape (N,) containing in-plane rotation angles (in radians).
    shiftX : torch.Tensor
        Tensor of shape (N,) containing X shifts.
    shiftY : torch.Tensor
        Tensor of shape (N,) containing Y shifts.
    batch_size : int, optional
        Number of particles processed per batch, by default 256.
    inplace : bool, optional
        If True, overwrites the input `particles` tensor to save memory.
        If False, allocates a new tensor for the aligned output.
        Default is True.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, H, W) containing the aligned particle images.

    Notes
    -----
    - Translations are applied using the Fourier shift theorem for
      subpixel accuracy.
    - Rotations are applied using `torch.nn.functional.grid_sample`.
    - The coordinate grid is defined in the range [-1, 1] with
      `align_corners=True`.
    """
    n, h, w = particles.shape

    # Initialize aligned images tensor
    if inplace:
        aligned = particles
    else:
        aligned = torch.empty_like(particles)

    # Process particles in batches
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)

        # Read batch data
        batch = particles[i:j]
        batch_shx = shiftX[i:j]
        batch_shy = shiftY[i:j]
        batch_ang = psi[i:j]

        # 1. Fourier shift
        shifted = fourier_shift_batch(batch, batch_shx, batch_shy)

        # 2. Rotation
        # Build rotation matrices: shape (B, 2, 2)
        cos = torch.cos(batch_ang)
        sin = torch.sin(batch_ang)
        zeros = torch.zeros_like(cos)

        rot_mats = torch.stack(
            [
                torch.stack([cos, -sin, zeros], dim=1),
                torch.stack([sin, cos, zeros], dim=1),
            ],
            dim=1,
        )

        # Build affine rotation grid for grid_sample
        grids = F.affine_grid(
            rot_mats, size=(shifted.size(0), 1, h, w), align_corners=True
        )

        # Prepare images for grid_sample -> (B, 1, h, w)
        imgs = shifted.unsqueeze(1)

        # Apply rotation through sampling
        rotated = F.grid_sample(
            imgs, grids, align_corners=True, padding_mode="zeros"
        ).squeeze(1)

        # Save rotated images to aligned tensor
        aligned[i:j] = rotated

    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="Applies alignment to particles from a .star file from  RELION 3.0+"
    )
    parser.add_argument("star", type=str, help="Path to the input .star file")
    parser.add_argument(
        "--out",
        type=str,
        default="aligned_particles.mrcs",
        help="Path to the output .mrcs file with the aligned particles",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite the output .mrcs file if it already exists"
    )
    args = parser.parse_args()

    particles, psi, shiftX, shiftY, pix_size = read_data(args.star, device=args.device)
    aligned = align_particles_batch_RELION(
        particles, psi, shiftX, shiftY, batch_size=256, inplace=True
    )

    mrcfile.write(args.out, data=aligned.detach().cpu().numpy(), overwrite=args.overwrite, voxel_size=pix_size)
    print(f"Aligned particles saved in {args.out}")


if __name__ == "__main__":
    main()
