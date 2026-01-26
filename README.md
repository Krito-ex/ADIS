# ADIS
## Aperture Diffraction Imaging Spectrometer
### open source of "Compact Snapshot Spectral Imaging with Calibration-Free Aperture Diffraction".

This open-source repository primarily contains the following:

- ADIS forward simulation, incorporating various masks, utilised the open-source code ‘Diffractio’;
- Performance of Different Algorithms on ADIS Under a Unified Simulation Benchmark;
- Reconstruction from measurements captured by the prototype cameras ADIS-V1 and ADIS-V2. (The real-world capture and reconstruction models discussed in this paper will soon be open-sourced);
- Visualization Methods for PSFs Across Different Spectral Bands and Reconstructed Spectral Images.







## 1. System Configuration and Forward Model:
<img src="./system.png"  height=300 width=900>
&nbsp;

- Install 'diffractio' package:

```shell
  pip install diffractio
```
- Run the following code to generate the PSFs for the ADIS corresponding to the physical parameters (28 uniformly distributed bands from 450 nm to 650 nm).
```shell
cd ./propagation/
  # Generate PSFs of ADIS with a resolution of 1024×1024×28
  python ADIS_PSF_1024.py --mask_period 200 --pixel_size 3.45 --unit_size 2
  # Generate PSFs of ADIS with a resolution of 512×512×28
  python ADIS_PSF_512.py --mask_period 200 --pixel_size 3.45 --unit_size 2
```




## 2. Simulation Experiments:
<img src="./simulation-results.jpg"  height=350 width=900>
&nbsp;

- Run the following code to compare the reconstruction accuracy of ADIS using different algorithms:

```shell
cd ./simulation/train_VST/

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template mst_plus_plus --method mst_plus_plus --outf ./exp/mst_plus_plus_diff/ --batch_size 8 --learning_rate 0.0004 --gpu_id 0  > ADIS_mst_plus_plus_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template restormer --method restormer --outf ./exp/restormer_diff/ --batch_size 4 --learning_rate 0.0004 --gpu_id 0,1  > ADIS_restormer_1024_bs4.log &

nohup python train_VST.py --psf_name PPSF/200um-1024.npy --template lambda_net --method lambda_net --outf ./exp/lambda_net_diff/ --batch_size 8 --learning_rate 0.00004 --gpu_id 0  > ADIS_lambda_net_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template unet --method unet --outf ./exp/unet_diff/ --batch_size 8 --learning_rate 0.0004 --gpu_id 0  > ADIS_unet_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template mirnet --method mirnet --outf ./exp/mirnet_diff/ --batch_size 8 --learning_rate 0.0001 --gpu_id 0,1  > ADIS_mirnet_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template tsa_net --method tsa_net --outf ./exp/tsa_net_diff/ --batch_size 8 --learning_rate 0.0004 --gpu_id 0,1  > ADIS_tsa_net_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template mprnet --method mprnet --outf ./exp/mprnet_diff/ --batch_size 8 --learning_rate 0.0004 --gpu_id 2  > ADIS_mprnet_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template CSST_9stg --method CSST_9stg --outf ./exp/CSST_9stg_diff/ --batch_size 8 --learning_rate 0.0004 --gpu_id 0,1,2,3  > ADIS_CSST_9stg_1024_bs8.log &

nohup python train_VST.py --psf_name PSF/200um-1024.npy --template ODAUVST_5stg --method ODAUVST_5stg --outf ./exp/ODAUVST_5stg_diff/ --batch_size 8 --learning_rate 0.0008 --gpu_id 0,1,2,3  > ADIS_ODAUVST_5stg_bs8.log &
```

- Run the following code to implement reconstruction on a high-resolution simulation scene:

```shell
cd ./simulation/test_VST/
```


## 3. Real Experiments:
<img src="./real-expriment.png"  height=480 width=900>
&nbsp;

- All data and models in this section will be open-sourced after the article is formally published.


## 4. Visualization:
<img src="./visualization.png"  height=400 width=900>
&nbsp;

```shell
cd ./visualization_tools_python
  # visualize reconstructed HSIs
  python hsi_render_rgb_dir.py
  # visualize reconstructed PSFs
  python psf_render_rgb.py
```


## 5. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
  pip install -r requirements.txt
```

## 6. Citation
If this repo helps you, please consider citing our works:


```shell
# ADIS+CSST ICCV 2023 Version
@inproceedings{lv2023aperture,
  title={Aperture Diffraction for Compact Snapshot Spectral Imaging},
  author={Lv, Tao and Ye, Hao and Yuan, Quan and Shi, Zhan and Wang, Yibo and Wang, Shuming and Cao, Xun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10574--10584},
  year={2023}
}
# ADIS+ODAUVST IEEE TPAMI Version

```
