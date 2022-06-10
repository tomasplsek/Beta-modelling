# Beta modelling

A jupyter notebook that simplifies beta modelling of X-ray images of eliptical galaxies. The notebook combines [Ipywidgets](https://github.com/jupyter-widgets/ipywidgets) with the [Sherpa 4.14](https://cxc.cfa.harvard.edu/sherpa4.14/) package and provides a simple graphical interface with a set models for fitting the 2D surface brightness distribution of images. The parameters of the model are easily adjusted, freezed, or tied using clickable sliders and boxes.

# Usage

The beta-modelling notebook can be used simply by running it in the Jupyter Notebook or Jupyterlab platform using Python environment with all required libraries (stated in Requirements). Alternatively, the notebook can be run using a [Voilà](https://github.com/voila-dashboards/voila) package:
```bash
$ voila beta_fitting.ipynb
```
which autoruns the whole notebook and displays the cell outputs in a new browser tab in a cleaner way than classical Jupyter notebook.

 The notebook automaticaly finds all fits files in current directory and lists them in the `Galaxy:` dropdown selection menu. When the galaxy image is loaded, one can pick a size scale of the fitted part of the image and also choose between various types of models (single or double beta model etc.) A given model is described by a set of parameters that can be adjusted, freezed, or tied with others using sliders and checkboxes.
 
 Whenever an optimal model and set of parameters is chosen by the user, it can be fitted using the `Fit` button. The fitted parameters can be saved into a text file using the `Save` button and the residual image is saved by the `Residual` button. Altarnatively, the user can run an MCMC simulation (`Run MCMC` button) to properly estimate the uncertainties of the fitted parameters and also correlations between them. One can set the length of the actual MCMC chain and also its the burned part as well as the fit statistics (chi2, cstat, etc.).

 The output window in the bottom right shows radial profile of both data and model (individual model components are displayed), original image, model image, and also a residual image obtained by substracting the model from the original image.

![](out.gif)

# Requirements

#### Python libraries:
`astropy`\
`corner`\
`ipywidgets`\
`matplotlib`\
`numpy`\
`pandas`\
`scipy`\
`sherpa`\
`voila` (optional - for clean output in new tab)

#### Data:
Processed X-ray (*Chandra*, XMM-Newton) image of eliptical galaxy \
    - background subtracted & exposure corrected if possible\
    - cropped and centered at the center of the galaxy\
    - excluded & filled point sources

# Example

The github repository includes three exemplary X-ray images of elliptical galaxies (NGC4649, NGC4778, NGC5813) observed by *Chandra X-ray observatory*. All observations were processed by classical CIAO procedures and removed for point sources using *dmfilth* routine.


# Todo

- add image preprocessing functionalities (finding and filling point sources)
- add other methods (unsharp masking, GGM)
- implement Arithmetic with X-ray images ([Churazov et al. 2016](https://arxiv.org/abs/1605.08999))
- implement [CADET](https://github.com/tomasplsek/CADET) predictions
- add cavity selection + significance estimation

