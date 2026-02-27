# Additional Results


### In-Vitro Phantom
In order to evaluate the generalizability of the proposed algorithm, it was applied to data acquired from a tissue-mimicking phantom (CIRS040) using a Philips S5-1 transducer without probe motion, creating a static scene. The temporal diffusion prior trained on the EchoNet-Dynamic dataset was then used for reconstruction. Given the lack of motion and the differences in spatial structure that exist between the CIRS phantom and EchoNet echocardiograms, this acquisition represents an out-of-distribution sample for the model. As with the in-house cardiac dataset, 90 focused lines and 11 diverging waves at the fundamental frequency were acquired for comparison. The acquisitions were beamformed and resized to a 90 x 112 grid to match the EchoNet-Dynamic prior. Contrast and resolution were then evaluated via gCNR and FWHM, respectively, for each reconstruction method, as shown in the figure below. The FWHM is calculated along a line in the image as the distance between the two points closest to the peak on either side at which the intensity equals half of the peak intensity (or, equivalently, -3dB from the peak). It is clear in the figure that the wire targets and inclusions are captured by the active perception strategy despite the significant domain shift, with improved recovery of the wire targets at depth compared to the equispaced and random strategies, as reflected by the improved FWHM result. While the main structural features are present in the active perception method, it is worth noting that the speckle resolution has visibly decreased, showing instead a smoother texture resembling those seen in EchoNet-Dynamic.

<img width="1106" height="775" alt="Screenshot 2026-01-14 at 16 07 02" src="https://github.com/user-attachments/assets/3267fc2c-a6d7-4fbc-a680-d9af85058cd8" />

### New Cardiac View (PLAX)
In order to evaluate the model's performance in a realistic out-of-distribution scenario, we benchmarked CASL with an EchoNet-Dynamic prior on target sequences from the [EchoNet-LVH dataset](https://echonet.github.io/lvh/), which constists of parasternal long-axis (PLAX) view echocardiograms, resized to 112x112 to match the input size expected by the EchoNet-Dynamic diffusion model. Because EchoNet-Dynamic contains only the apical 4-chamber view, these PLAX view targets represent out-of-distribution samples. We ran the model on 10 sequences from different patients, each containing 100 frames. The following shows the reconstruction accuracy, as well as an example cine-loop of the reconstructed video using only 7/112 scan lines.

<img width="1050" height="750" alt="LVH_lpips_psnr" src="https://github.com/user-attachments/assets/14bdd674-43f0-4c95-bc10-e8aa9dc193df" />

![target_reconstruction_plax_7_lines](https://github.com/user-attachments/assets/8997aae1-119e-45a6-9216-760c3b9a23a9)


### Robustness across patients

An essential feature of any image reconstruction method in medical imaging is robustness against outliers, ensuring that the performance is consistent across patients.
In order to evaluate this in our approach, we ran active perception on the first 100 frames of each of the 500 sequences in the unseen EchoNet-Dynamic test set, with a measurement budget of 14 lines per frame.
In the figure below, we plot the reconstruction quality as measured by PSNR against the ejection fraction of each patient, examining the correlation between the two. 
The figure shows that the reconstruction quality is independent of the patient’s ejection fraction, indicating a lack of bias against outlier patients.

<img width="1050" height="750" alt="ef_psnr_correlation_cogntive_ultrasound" src="https://github.com/user-attachments/assets/54d20903-d139-4234-a8b4-e27bdcdd3b17" />

### SeqDiff

The following figure shows the relation of the number of diffusion steps to the reconstruction quality in terms of PSNR for regular and [SeqDiff](https://ieeexplore.ieee.org/document/10889752/), when using our method, which motivates employing SeqDiff for enhanced reconstruction quality and speed.

<img width="1050" height="750" alt="seqdiff_vs_regular_diffusion" src="https://github.com/user-attachments/assets/21307368-a0ef-4707-967c-4fb81834ad0c" />

### Reconstruction Strategy Comparison: _First Particle_ versus _Posterior Mean_
Here we show the effect of increasing $N_p$ on reconstruction metrics for both methods of determining a point estimate reconstruction to represent the posterior distribution. The methods are (i) choosing the first particle from the distribution, resulting effectively in a random choice of particle, labelled _Choose First_, and (ii) computing the sample posterior mean from the entire set of particles, labelled _Posterior Mean_. It is clear from the results that _Choose First_ generally achieves improved perceptual loss, at the expense of an increased distortion, both of which are invariant to $N_p$. _Posterior Mean_, on the other hand, is affected by $N_p$, with a slight improvement in distortion but disimprovement in perceptual similarity as $N_p$ increases, demonstrating the perception-distortion trade-off as the reconstruction becomes blurred by averaging across additional particles.

<img width="1038" height="757" alt="choose_first_vs_posterior_mean" src="https://github.com/user-attachments/assets/537a1ffa-be34-4c3e-a5a0-904632ff8eab" />
