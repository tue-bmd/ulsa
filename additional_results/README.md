# Additional Results


### In-Vitro Phantom
In order to evaluate the generalizability of the proposed algorithm, it was applied to data acquired from a tissue-mimicking phantom (CIRS040) using a Philips S5-1 transducer without probe motion, creating a static scene. The temporal diffusion prior trained on the EchoNet-Dynamic dataset was then used for reconstruction. Given the lack of motion and the differences in spatial structure that exist between the CIRS phantom and EchoNet echocardiograms, this acquisition represents an out-of-distribution sample for the model. As with the in-house cardiac dataset, 90 focused lines and 11 diverging waves at the fundamental frequency were acquired for comparison. The acquisitions were beamformed and resized to a 90 x 112 grid to match the EchoNet-Dynamic prior. Contrast and resolution were then evaluated via gCNR and FWHM, respectively, for each reconstruction method, as shown in the figure below. The FWHM is calculated along a line in the image as the distance between the two points closest to the peak on either side at which the intensity equals half of the peak intensity (or, equivalently, -3dB from the peak). It is clear in the figure that the wire targets and inclusions are captured by the active perception strategy despite the significant domain shift, with improved recovery of the wire targets at depth compared to the equispaced and random strategies, as reflected by the improved FWHM result. While the main structural features are present in the active perception method, it is worth noting that the speckle resolution has visibly decreased, showing instead a smoother texture resembling those seen in EchoNet-Dynamic.

<img width="1106" height="775" alt="Screenshot 2026-01-14 at 16 07 02" src="https://github.com/user-attachments/assets/3267fc2c-a6d7-4fbc-a680-d9af85058cd8" />

### Robustness across patients

An essential feature of any image reconstruction method in medical imaging is robustness against outliers, ensuring that the performance is consistent across patients.
In order to evaluate this in our approach, we ran active perception on the first 100 frames of each of the 500 sequences in the unseen EchoNet-Dynamic test set, with a measurement budget of 14 lines per frame.
In the figure below, we plot the reconstruction quality as measured by PSNR against the ejection fraction of each patient, examining the correlation between the two. 
The figure shows that the reconstruction quality is independent of the patient’s ejection fraction, indicating a lack of bias against outlier patients.

<img width="1050" height="750" alt="ef_psnr_correlation_cogntive_ultrasound" src="https://github.com/user-attachments/assets/54d20903-d139-4234-a8b4-e27bdcdd3b17" />
