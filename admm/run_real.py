from admm import admm_scheme
from weights import *
import argparse
import mrcfile
import napari


def main():
    parser = argparse.ArgumentParser(description="Run ADMM iterations on given data")
    parser.add_argument("input_file", type=str, help="Path to data")
    args = parser.parse_args()

    images = torch.tensor(mrcfile.read(args.input_file))
    fourier_images = torch.fft.rfft2(images, norm="ortho")

    initial_ref_real = torch.mean(images, dim=0)
    initial_ref_fourier = torch.mean(fourier_images, dim=0)

    weight_function_real = get_weight_function("huber", params={"delta": 1.4})
    weight_function_fourier = get_weight_function("huber", params={"delta": 1.4})

    converged, iters, estimation_real, estimation_fourier = admm_scheme(
        images,
        fourier_images,
        ctf=torch.tensor(1),
        initial_ref_real=initial_ref_real,
        initial_ref_fourier=initial_ref_fourier,
        mu=1.0,
        C=1.0,
        weight_function_real=weight_function_real,
        weight_function_fourier=weight_function_fourier,
        max_iter=50,
    )
    print(f"{converged = }")
    print(f"{iters = }")

    viewer = napari.Viewer()
    viewer.add_image(images[:50], name="50 first images")
    viewer.add_image(torch.mean(images, dim=0), name="Average of all images")
    viewer.add_image(estimation_real, name="Estimation with ADMM")
    napari.run()



if __name__ == "__main__":
    main()
