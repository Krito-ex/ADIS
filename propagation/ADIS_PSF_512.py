import os
import cv2
from diffractio import degrees, eps, mm, no_date, np, um, nm
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_fields_XZ import Scalar_field_XZ
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
import matplotlib.pyplot as plt
import time

import argparse
parser = argparse.ArgumentParser(description="Hardware parameters of A-ADIS")
parser.add_argument('--focal_length', type=float, default=50, help='mm, focal length of main lens')
parser.add_argument("--mask_period", type=float, default=200, help='um, period of mask')
parser.add_argument("--fill_ratio", type=float, default=0.1, help='mask duty ratio')
parser.add_argument("--focused_depth_center", type=float, default=100000, help='mm, focused depth center')
parser.add_argument("--pixel_size", type=float, default=3.45, help='um, focused depth center')
parser.add_argument("--unit_size", type=float, default=2, help='um, simulated unit size')
parser.add_argument("--nC", type=int, default=28, help='number of wavelengths')

# Calculation parameter
parser.add_argument("--gpu_id", type=str, default='2,3')
opt = parser.parse_args()



def resize(img, target_len):
    '''
    :param line: [len,len]
    :param target_len: int
    :return: target_line: [target_len, target_len]
    '''
    [_, current_len] = img.shape
    resized_img = cv2.resize(img, (target_len, target_len), interpolation=cv2.INTER_CUBIC)
    return resized_img


def calculate_point_depth_psf(point_depth, args):
    '''
    :param point_depth: [1]
    :param args: hardware parameter
    :return: psf, intensity of x axis [nC 1 512]
    '''

    fl = args.focal_length * mm
    fdc = args.focused_depth_center * mm
    pixel_size = args.pixel_size * um
    mask_period = args.mask_period * um
    fill_factor = args.fill_ratio
    unit_size = args.unit_size * um
    nC = args.nC

    psfs = np.zeros((nC, 520, 520), dtype=np.float32)

    z_source = point_depth * mm
    aperture = 10 * mm

    # 3.45×520×2 pixel; 897×2×2um unit
    numdataX = 897
    numdataY = 897
    length_X = numdataX * unit_size
    length_Y = numdataY * unit_size

    x = np.linspace(-length_X / 2, length_X / 2, numdataX)
    y = np.linspace(-length_Y / 2, length_Y / 2, numdataY)

    # propagation_length = 1 / (1 / fl - 1 / fdc)
    propagation_length = fl
    wavelengths = (np.linspace(450, 650, (nC+1))+(650-450)/(2*nC)) * nm
    for i in range(len(wavelengths)-1):
        wavelength = wavelengths[i]
        print(i, wavelength)
        u1 = Scalar_source_XY(x=x, y=y, wavelength=wavelength)
        u1.spherical_wave(A=1, r0=(0 * um, 0 * um), z0=-z_source, radius=aperture / 2, normalize=False)
        u2 = Scalar_mask_XY(x=x, y=y, wavelength=wavelength)
        u2.lens(r0=(0 * um, 0 * um), focal=fl, radius=(aperture / 2, aperture / 2), angle=0 * degrees)
        u3 = Scalar_mask_XY(x, y, wavelength)
        u3.grating_2D(r0=(0,0), period=mask_period, a_min=0, a_max=1, fill_factor=fill_factor)
        t1 = u1 * u2 * u3
        u = t1.RS(z=propagation_length)
        field = np.abs(u.u) / np.max(np.abs(u.u))
        psf = field ** 2
        psf = resize(psf, target_len=int((numdataX * unit_size)/pixel_size))
        psfs[i] = psf
    return psfs[:,4:516,4:516]     # 512


if __name__ == '__main__':
    start_time = time.time()
    depth = opt.focused_depth_center
    psfs = calculate_point_depth_psf(point_depth=depth, args=opt)
    end_time = time.time()
    print(end_time - start_time)
    print(psfs.max())

    save_root = '/home/data1/lvtao/Meta-adis-amax-151/VST-SDI/ADIS_tools_new/PSF/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = save_root + f'{opt.mask_period}um-512.npy'
    np.save(save_path, psfs)
    
