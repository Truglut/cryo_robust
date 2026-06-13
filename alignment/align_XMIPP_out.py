"""
Alignment code for particles whose alignment parameters have been calculated by XMIPP

Original author: Erney Ramírez Aportela
The code has been modified by Andrés Contreras Santos to split data reading and
alignment into two different functions.
"""

import torch
import argparse
import mrcfile
import torch.nn.functional as F

from align_RL_out import read_data, fourier_shift_batch

@torch.no_grad()
def align_particles_batch_XMIPP(
    particles: torch.Tensor,
    angles_rad: torch.Tensor,
    shiftX: torch.Tensor,
    shiftY: torch.Tensor,
    batch_size: int = 256,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Aligns a set of particles using batched Fourier shifts and spatial rotations.
    Follows XMIPP's conventions for alignment.

    The alignment consists of:
    1. In-plane rotations applied via grid sampling.
    2. Subpixel translations applied in Fourier space.

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
        batch_ang = angles_rad[i:j]

        # 1. Rotation
        cos = torch.cos(batch_ang)
        sin = torch.sin(batch_ang)
        zeros = torch.zeros_like(cos)

        # Build affine rotation matrices: shape(B, 2, 3)
        rot_mats = torch.stack(
            [
                torch.stack([cos, -sin, zeros], dim=1),
                torch.stack([sin, cos, zeros], dim=1)
            ]
        )

        # Build affine rotation grid for grid_sample
        grids = F.affine_grid(
            rot_mats, size=(batch.size(0), 1, h, 2), align_corners=True
        )

        # Prepare images for grid_sample -> (B, 1, h, w)
        imgs = batch.unsqueeze(1)

        # Apply rotation through sampling
        rotated = F.grid_sample(
            imgs, grids, align_corners=True, padding_mode="zeros"
        )

        # 2. Fourier shift
        shifted = fourier_shift_batch(rotated, batch_shx, batch_shy)

        # Save aligned images to aligned tensor
        aligned[i:j] = shifted

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
        help="Overwrite the output .mrcs file if it already exists",
    )
    args = parser.parse_args()

    particles, psi, shiftX, shiftY, pix_size = read_data(args.star, device=args.device)
    aligned = align_particles_batch_XMIPP(
        particles, psi, shiftX, shiftY, batch_size=256, inplace=True
    )

    mrcfile.write(
        args.out,
        data=aligned.detach().cpu().numpy(),
        overwrite=args.overwrite,
        voxel_size=pix_size,
    )
    print(f"Aligned particles saved in {args.out}")


if __name__ == "__main__":
    main()
