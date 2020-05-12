# Multi-task Learning for End-to-end Noise-robust Bandwidth Extension

For demo please refer https://nanahou.github.io/mtl-mbe/

The codes here are for noise-robust bandwidth extension. 
Bandwidth extension aims to reconstruct wideband speech signals from narrowband inputs to improve speech quality. The prior work mostly performs bandwidth extension under ideal conditions, which assumes that the narrowband signals are clean without noise. The use of such extension techniques is greatly limited in practice when signals are corrupted by noise. 
To alleviate such problem, we propose an end-to-end time-domain framework for noise-robust bandwidth extension, that jointly optimizes mask-based speech enhancement and the ideal bandwidth extension module with multi-task learning. The proposed framework avoids decomposing the signals into magnitude and phase spectrums, and therefore requires no phase estimation. Experimental results show that the proposed method achieves 14.3% and 15.8% relative improvements over the best baseline UEE in terms of perceptual evaluation of speech quality (PESQ) and log-spectral distortion (LSD), respectively. Furthermore, our method is 3 times more compact than UEE in terms of the number of parameters.
