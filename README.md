# Immune Cell Migration Analysis

The repository holds code for analysing cell migration using a biased-persistent random walk as well as attractant dynamics.
This repo started out by trying to reproduce the computational methods in Weavers, Liepe, et al. "Systems Analysis of the Dynamic Inflammatory Response to Tissue Damage Reveals Spatiotemporal Properties of the Wound Attractant Gradient. Curr. Biol. 2016;26(15):1975–1989. http://dx.doi.org/10.1016/j.cub.2016.06.012. Here we expand upon that functionality. For a stable version, please refer to the repository that this is forked from: https://github.com/nickelnine37/DrosophilaWoundAnalysis

The requirements for this project are Numpy, Pandas, Scipy, Matplotlib, Skimage, tqdm and Jupyter. It has been tested with the following versions:

```
Python version:     3.7.3
Numpy version:      1.16.4
Matplotlib version: 3.0.3
Pandas version:     0.24.2
Skimage version:    0.15.0
Scipy version:      1.2.1
tqdm version:       4.32.1
Jupyter version:    4.4.0
```

but *should* work with python 3 and any other set of compatible package versions.

In addition, in order to input mp4 files, the module `ffmpeg-python` is needed. This can be installed by running

```
pip install ffmpeg-python
```

Start in the Notebooks folder!
