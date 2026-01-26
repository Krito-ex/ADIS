import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
from architecture import *
from fvcore.nn import FlopCountAnalysis

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand'] / 65536.
        elif "HSI" in img_dict:
            img = img_dict['HSI']
        else:
            raise AttributeError('Uknown data temp')
        img = np.array(img).astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    # test_data = np.zeros((len(scene_list), 926, 926, 28))
    test_data = np.zeros((len(scene_list), 586, 586, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img_dict = sio.loadmat(scene_path)
        if 'HSI_crop_926' in img_dict:
            img = img_dict['HSI_crop_926'] / np.max(img_dict['HSI_crop_926'])
        if 'HSI_crop_586' in img_dict:
            img = img_dict['HSI_crop_586'] / np.max(img_dict['HSI_crop_586'])
        elif 'HSI_crop_256' in img_dict:
            img = img_dict['HSI_crop_256'] / np.max(img_dict['HSI_crop_256'])
        test_data[i, :, :, :] = img
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def init_filter_function(size):
    filter_function = np.ones((28, size, size), dtype=np.float32)
    para_B = 0.01 * np.array(
        [41.329, 42.693, 42.571, 42.449, 41.015, 40.574, 38.157, 33.801, 31.558, 29.606, 25.58, 18.842, 15.388,
         11.558, 8.802, 5.814, 4.458, 2.367, 2.258, 1.261, 1.09, 0.751, 0.565, 0.413, 0.265, 0.246, 0.462, 0.582], dtype=np.float32)
    para_G = 0.01 * np.array(
        [8.8, 10.316, 10.9, 11.989, 14.897, 15.848, 19.707, 21.605, 22.494, 23.652, 27.439, 34.137, 38.332, 42.359,
         45.631, 47.122, 47.225, 43.701, 43.264, 35.983, 33.943, 27.93, 21.888, 14.946, 7.796, 5.055, 3.428, 3.289], dtype=np.float32)
    para_R = 0.01 * np.array(
        [1.852, 1.716, 1.766, 1.823, 1.922, 1.913, 1.738, 1.605, 1.586, 1.593, 1.712, 2.092, 2.426, 2.93, 3.401,
         3.6, 3.185, 1.542, 1.419, 4.222, 5.614, 20.869, 32.125, 38.664, 39.5, 38.008, 35.146, 34.333], dtype=np.float32)
    filter_function[:, 0::2, 0::2] *= para_G.reshape(-1, 1, 1)
    filter_function[:, 0::2, 1::2] *= para_B.reshape(-1, 1, 1)
    filter_function[:, 1::2, 0::2] *= para_R.reshape(-1, 1, 1)
    filter_function[:, 1::2, 1::2] *= para_G.reshape(-1, 1, 1)
    return filter_function

def init_psf(psf_path):
    psf = np.load(psf_path).astype(np.float32)
    [C, h, w] = psf.shape
    for i in range(C):
        psf[i] = psf[i] / (np.sum(psf[i]))
    return psf

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def init_input_adis(test_gt, filter_function, psf, crop_size):
    [bs, C, YH, YW] = test_gt.shape     # 10, 28 , 586, 586
    [C, h, w] = filter_function.shape

    offset1 = (YH - crop_size) // 2
    input_cube = test_gt[:, :, offset1:offset1+crop_size, offset1:offset1+crop_size]

    spa_offset = (crop_size - h) // 2
    gt_batch = input_cube[:, :, spa_offset:spa_offset+h, spa_offset:spa_offset+w]

    ff_batch = np.repeat(filter_function[np.newaxis, :, :, :], bs, axis=0)
    ff_batch = torch.FloatTensor(ff_batch)

    psf_batch = np.repeat(psf[np.newaxis, :, :, :], bs, axis=0)
    psf_batch = torch.FloatTensor(psf_batch)

    input_cube = input_cube.cuda().float()
    gt_batch = gt_batch.cuda().float()
    ff_batch = ff_batch.cuda().float()
    psf_batch = psf_batch.cuda().float()

    input_mea = Measurement_Render(input_cube, ff_batch, psf_batch)
    return input_mea, gt_batch, ff_batch, psf_batch


def Fast_rFFT2d_GPU_batch(DHSI, psf_cube):
    '''
    Args:
        DHSI: [bs, C, H, W]
        psf_cube: [bs, C, h, w]
    Returns:
        output_cube: [bs, C, H, W]
    not crop
    '''
    [bs,C,H,W] = DHSI.shape
    [bs,C,h,w] = psf_cube.shape
    if h<1024 or w<1024:   # 保证psf在512分辨率和1024分辨率进行fft运算时，有一致的运算空间
        PSF = torch.zeros((bs, C, 1024, 1024), dtype=psf_cube.dtype, device=psf_cube.device)
        PSF[:, :, (1024 - h) // 2:(1024 - h) // 2 + h, (1024 - w) // 2:(1024 - w) // 2 + w] = psf_cube
        psf_cube = PSF
    [bs, C, h, w] = psf_cube.shape
    output_cube = torch.zeros([bs, C, H, W], device=DHSI.device)
    # prepare padding size to accommodate FFT-based convolution
    YH = H + h - 1
    YW = W + w - 1
    YH = 2 ** (int(torch.log2(torch.tensor(YH)))+1)
    YW = 2 ** (int(torch.log2(torch.tensor(YW)))+1)
    for i in range(bs):
        HSI = DHSI[i]   # [C,H,W]
        psf = psf_cube[i]   # [C,h,w]
        HSI_padded = torch.zeros([C, YH, YW], device=HSI.device)
        psf_padded = torch.zeros([C, YH, YW], device=psf.device)
        HSI_padded[:, (YH - H) // 2:(YH - H) // 2 + H, (YW - W) // 2:(YW - W) // 2 + W] = HSI
        psf_padded[:, (YH - h) // 2:(YH - h) // 2 + h, (YW - w) // 2:(YW - w) // 2 + w] = psf
        HSI_fft = torch.fft.rfft2(HSI_padded, s=(YH,YW)).cuda()
        psf_fft = torch.fft.rfft2(psf_padded, s=(YH,YW)).cuda()
        # Element-wise multiplication in the frequency domain
        conv_result_fft = HSI_fft * psf_fft
        # inverse FFT to get the convolution result in the spatial domain
        conv_result = torch.fft.irfft2(conv_result_fft)[:,:YH,:YW]
        # roll
        conv_result = torch.roll(conv_result, shifts=(YH//2, YW//2), dims=(1,2))
        # crop the result to the valid region
        meas = conv_result[:, YH//2-H//2:YH//2+H//2, YW//2-W//2:YW//2+W//2]
        output_cube[i] = meas.squeeze(0)
    return output_cube


def add_gaussian_noise(measurements, sigma):
    noise = torch.normal(mean=0.0, std=sigma, size=measurements.shape, device=measurements.device)
    noisy_measurements = measurements + noise
    return noisy_measurements

add_noise = False
def Measurement_Render(input_cube, ff_batch, psf_batch):
    conv_cube = Fast_rFFT2d_GPU_batch(input_cube, psf_batch)
    size = ff_batch.shape[2]
    crop_size = input_cube.shape[2]
    spa_offset = (crop_size - size) // 2
    filtered_cube = conv_cube[:, :, spa_offset:spa_offset+size, spa_offset:spa_offset+size]
    filtered_cube = filtered_cube * ff_batch
    Measurement = torch.sum(filtered_cube, dim=1, keepdim=True)
    Measurement = Measurement / float(input_cube.shape[1]) * 15  # 4 for DOE    15 for ADIS   6 for diffuser
    if add_noise:
        Measurement = add_gaussian_noise(Measurement, 0.02)
    return Measurement

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def Net_para_calculate(test_model, H = 256, W = 256, C = 28, N = 1):
    model = test_model.cuda()
    print(model)
    input_cube=None
    # input_cube = torch.randn((N, C, 512, 512)).cuda()
    ff_batch = torch.randn((N, C, 256, 256)).cuda()
    psf_batch = torch.randn((N, C, 1024, 1024)).cuda()
    y = torch.randn((N,1,256,256)).cuda()
    # y=None

    flops = FlopCountAnalysis(model,(input_cube, ff_batch, psf_batch, y))
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    # model, FDL_loss = model_generator('tsa_net', '/home/root/data1/lvtao/CSST/D-simulation/train_code_D_ADIS/exp/tsa_net/2023_04_20_02_10_24/model/model_epoch_211.pth')

    data_root = '/home/data1/lvtao/Meta-adis-amax-151/VST-SDI/simulation/PAMI_backup/different_method/'
    data_name = 'ADIS_ODAUVST_model_epoch_243.pth'
    data_path = os.path.join(data_root, data_name)
    data_path = '/home/data1/lvtao/Meta-adis-amax-151/VST-SDI/simulation/train_VST_final/exp/W-DDPUT_5stg_diff/2025_03_27_12_39_09/model/model_epoch_274.pth'
    model = model_generator('DDPUT_5stg', pretrained_model_path=data_path)

    Net_para_calculate(model)

