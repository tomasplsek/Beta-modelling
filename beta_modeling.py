#!/usr/bin/env python3
"""
Beta Profile Model Fitting Module
=================================

This module handles data loading, beta model fitting using Sherpa, and provides
structured data for plotting. All Sherpa-specific functionality is isolated here.

Author: Extracted from beta_fitting_improved.py
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

# Import utility functions
from utils import suppress_astropy_warnings, calculate_pixel_scale_from_wcs
suppress_astropy_warnings()

from astropy.wcs import WCS
WCS._do_not_fail_on_wcs_error = True

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D, CCDData

# Sherpa imports
from sherpa.astro.ui import *
from sherpa.plot import *
from sherpa_contrib.profiles import *


@dataclass
class FittingConfig:
    """Configuration class for beta fitting parameters."""
    
    # Statistical parameters
    quantiles: Tuple[float, float] = (0.15865525, 0.84134475)
    
    # Model options
    available_models: List[str] = None
    available_methods: List[str] = None
    available_statistics: List[str] = None
    available_scales: List[str] = None
    
    # Default values
    default_method: str = "levmar"
    default_statistic: str = "cstat"
    default_mcmc_length: int = 1000
    default_burn_in: int = 100
    
    # Image processing
    pixel_scale: float = 0.492  # arcsec per pixel
    smoothing_map: Dict[int, int] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                'Single beta', 'Single beta + gauss', 'Double beta',
                'Double beta + gauss', 'Triple beta', 'Four beta', 'Five beta'
            ]
        
        if self.available_methods is None:
            self.available_methods = ["levmar", "neldermead", "moncar"]
        
        if self.available_statistics is None:
            self.available_statistics = ["cstat", "cash", "chi2", "chi2xspecvar"]
        
        if self.available_scales is None:
            self.available_scales = [
                "original", "128 x 128 pixels", "256 x 256 pixels",
                "384 x 384 pixels", "512 x 512 pixels", "640 x 640 pixels",
                "768 x 768 pixels"
            ]
        
        if self.smoothing_map is None:
            self.smoothing_map = {
                64: 1, 128: 1, 192: 1, 256: 1.5, 384: 2,
                512: 3, 640: 4, 768: 4, 896: 5
            }


@dataclass
class RadialProfileData:
    """Data class for radial profile information."""
    radius_pixels: np.ndarray
    radius_arcsec: np.ndarray
    data: np.ndarray
    data_error: np.ndarray
    model: np.ndarray
    components: Dict[str, np.ndarray]
    pixel_scale: float
    
    @property
    def radius(self) -> np.ndarray:
        """For backward compatibility, return pixel radius (primary axis)."""
        return self.radius_pixels
    
    def get_residuals(self) -> np.ndarray:
        """Calculate residuals in sigma units."""
        return (self.data - self.model) / self.data_error


@dataclass
class ImageData:
    """Data class for image information."""
    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    center: Tuple[float, float]
    size: float
    smoothing_std: float
    wcs: Optional[WCS] = None
    
    @property
    def data_cutout(self) -> np.ndarray:
        """Get data cutout around center."""
        x0, y0 = self.center
        return self.data[int(x0-self.size):int(x0+self.size), 
                        int(y0-self.size):int(y0+self.size)]
    
    @property
    def model_cutout(self) -> np.ndarray:
        """Get model cutout around center."""
        x0, y0 = self.center
        return self.model[int(x0-self.size):int(x0+self.size), 
                         int(y0-self.size):int(y0+self.size)]
    
    @property
    def residual_cutout(self) -> np.ndarray:
        """Get residual cutout around center."""
        x0, y0 = self.center
        return self.residual[int(x0-self.size):int(x0+self.size), 
                            int(y0-self.size):int(y0+self.size)]
    
    def get_pixel_scale_arcsec(self) -> float:
        """Calculate pixel scale in arcseconds from WCS information."""
        return calculate_pixel_scale_from_wcs(self.wcs)


class ModelParameter:
    """Wrapper class for model parameters with validation."""
    
    def __init__(self, name: str, value: float, min_val: float, max_val: float,
                 frozen: bool = False, linked: bool = False):
        self.name = name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.frozen = frozen
        self.linked = linked
    
    def validate(self) -> bool:
        """Validate parameter value is within bounds."""
        return self.min_val <= self.value <= self.max_val
    
    def __repr__(self) -> str:
        status = "frozen" if self.frozen else "free"
        if self.linked:
            status += "+linked"
        return f"{self.name}: {self.value:.3f} [{self.min_val}, {self.max_val}] ({status})"


class GalaxyImage:
    """Class to handle galaxy image loading and processing."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.name = self.filepath.stem
        self._data = None
        self._header = None
        self._wcs = None
        self._shape = None
        self._center = None
    
    @property
    def data(self) -> np.ndarray:
        """Lazy loading of image data."""
        if self._data is None:
            self._load_image()
        return self._data
    
    @property
    def header(self) -> fits.Header:
        """Image header."""
        if self._header is None:
            self._load_image()
        return self._header
    
    @property
    def wcs(self) -> WCS:
        """World coordinate system."""
        if self._wcs is None:
            self._wcs = WCS(self.header)
        return self._wcs
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape."""
        if self._shape is None:
            self._shape = self.data.shape
        return self._shape
    
    @property
    def center(self) -> Tuple[float, float]:
        """Image center coordinates."""
        if self._center is None:
            self._center = (self.shape[0] / 2, self.shape[1] / 2)
        return self._center
    
    def _load_image(self):
        """Load the FITS image."""
        try:
            with fits.open(self.filepath) as hdul:
                self._data = hdul[0].data
                self._header = hdul[0].header.copy()
        except Exception as e:
            raise IOError(f"Failed to load image {self.filepath}: {e}")
    
    def get_cutout(self, size: Union[str, int]) -> np.ndarray:
        """Get a cutout of the image."""
        if size == "original":
            return self.data
        
        if isinstance(size, str):
            size_val = int(size.split()[0].split("x")[0])
        else:
            size_val = size
        
        center_x, center_y = self.center
        if size_val > center_x * 2:
            raise ValueError(f"Size {size_val} is larger than image size")
        
        x_start = int(center_x - size_val // 2)
        x_end = int(center_x + size_val // 2)
        y_start = int(center_y - size_val // 2)
        y_end = int(center_y + size_val // 2)
        
        return self.data[x_start:x_end, y_start:y_end]


class BetaModel:
    """Class to handle beta model setup and parameter management."""
    
    def __init__(self, model_type: str, image_center: Tuple[float, float]):
        self.model_type = model_type
        self.image_center = image_center
        self.parameters = {}
        self._setup_model()
    
    def _setup_model(self):
        """Setup the sherpa model based on model type."""
        reset()
        clean()
        
        if self.model_type == "Single beta":
            set_source(beta2d.b1 + const2d.bkg)
        elif self.model_type == "Single beta + gauss":
            set_source(beta2d.b1 + gauss2d.g1 + const2d.bkg)
        elif self.model_type == "Double beta":
            set_source(beta2d.b1 + beta2d.b2 + const2d.bkg)
        elif self.model_type == "Double beta + gauss":
            set_source(beta2d.b1 + beta2d.b2 + gauss2d.g1 + const2d.bkg)
        elif self.model_type == "Triple beta":
            set_source(beta2d.b1 + beta2d.b2 + beta2d.b3 + const2d.bkg)
        elif self.model_type == "Four beta":
            set_source(beta2d.b1 + beta2d.b2 + beta2d.b3 + beta2d.b4 + const2d.bkg)
        elif self.model_type == "Five beta":
            set_source(beta2d.b1 + beta2d.b2 + beta2d.b3 + beta2d.b4 + beta2d.b5 + const2d.bkg)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self._setup_initial_parameters()
    
    def _setup_initial_parameters(self):
        """Setup initial parameter values and bounds."""
        x0, y0 = self.image_center
        
        # Primary beta component
        set_par(b1.xpos, val=x0, min=0, max=2*x0)
        set_par(b1.ypos, val=y0, min=0, max=2*y0)
        set_par(b1.ellip, val=0, min=0, max=0.7)
        set_par(b1.theta, val=0)
        set_par(b1.r0, val=1, min=5e-2, max=1e3)
        set_par(b1.ampl, val=50, min=1e-2, max=5e4)
        set_par(b1.alpha, val=1, min=0.1, max=10)
        
        # Secondary beta component (if applicable)
        if "Double" in self.model_type or "Triple" in self.model_type or "Four" in self.model_type or "Five" in self.model_type:
            set_par(b2.xpos, val=x0, min=0, max=2*x0)
            set_par(b2.ypos, val=y0, min=0, max=2*y0)
            set_par(b2.ellip, val=0, min=0, max=0.7)
            set_par(b2.theta, val=0)
            set_par(b2.r0, val=20, min=1e-1, max=1e3)
            set_par(b2.ampl, val=20, min=1e-2, max=5e4)
            set_par(b2.alpha, val=1, min=0.1, max=10)
        
        # Tertiary beta component (if applicable)
        if "Triple" in self.model_type or "Four" in self.model_type or "Five" in self.model_type:
            set_par(b3.xpos, val=x0, min=0, max=2*x0)
            set_par(b3.ypos, val=y0, min=0, max=2*y0)
            set_par(b3.ellip, val=0, min=0, max=0.7)
            set_par(b3.theta, val=0)
            set_par(b3.r0, val=100, min=1, max=5e3)
            set_par(b3.ampl, val=1, min=1e-2, max=5e4)
            set_par(b3.alpha, val=1, min=0.1, max=10)
        
        # Quaternary beta component (if applicable)
        if "Four" in self.model_type or "Five" in self.model_type:
            set_par(b4.xpos, val=x0, min=0, max=2*x0)
            set_par(b4.ypos, val=y0, min=0, max=2*y0)
            set_par(b4.ellip, val=0, min=0, max=0.7)
            set_par(b4.theta, val=0)
            set_par(b4.r0, val=150, min=1, max=5e3)
            set_par(b4.ampl, val=1, min=1e-2, max=5e4)
            set_par(b4.alpha, val=1, min=0.1, max=10)
        
        # Quinary beta component (if applicable)
        if "Five" in self.model_type:
            set_par(b5.xpos, val=x0, min=0, max=2*x0)
            set_par(b5.ypos, val=y0, min=0, max=2*y0)
            set_par(b5.ellip, val=0, min=0, max=0.7)
            set_par(b5.theta, val=0)
            set_par(b5.r0, val=200, min=1, max=5e3)
            set_par(b5.ampl, val=1, min=1e-2, max=5e4)
            set_par(b5.alpha, val=1, min=0.1, max=10)
        # Gaussian component (if applicable)
        if "gauss" in self.model_type:
            set_par(g1.xpos, val=x0, min=x0-40, max=x0+40)
            set_par(g1.ypos, val=y0, min=y0-40, max=y0+40)
            set_par(g1.ellip, val=0, min=0, max=0.7)
            set_par(g1.theta, val=0)
            set_par(g1.fwhm, val=1, min=1e-3, max=2e1)
            set_par(g1.ampl, val=50, min=1e-3, max=1e5)
        
        # Background
        set_par(bkg.c0, val=1e-3, min=0.001, max=200)


class MCMCAnalyzer:
    """Handle MCMC analysis and results."""
    
    def __init__(self, config: FittingConfig):
        self.config = config
    
    def run_mcmc(self, length: int, burn_in: int, galaxy_name: str, model_type: str = None, return_stats: bool = False) -> pd.DataFrame:
        """Run MCMC analysis and return results."""
        # Setup covariance
        covar()
        res = get_covar_results()
        
        # Setup sampler
        set_sampler("metropolismh")
        set_sampler_opt('defaultprior', True)
        
        print(f"\nRunning MCMC simulation with {length} iterations.")
        print(f"First {burn_in} iterations will be burned.\n")
        
        # Run MCMC
        stats, accept, params = get_draws(1, niter=length)
        
        # Calculate acceptance rate before burn-in removal
        total_samples = len(accept)
        accepted_samples = np.sum(accept)
        acceptance_rate = (accepted_samples / total_samples) * 100
        
        print(f"Total samples: {total_samples}")
        print(f"Accepted samples: {accepted_samples} ({acceptance_rate:.1f}%)")
        
        # Apply burn-in
        stats_burned = stats[burn_in:]
        accept_burned = accept[burn_in:]
        params_burned = params[:, burn_in:]
        
        # Filter only accepted samples after burn-in
        accepted_mask = accept_burned.astype(bool)
        stats_accepted = stats_burned[accepted_mask]
        params_accepted = params_burned[:, accepted_mask]
        
        # Report post-burn-in acceptance
        post_burnin_total = len(accept_burned)
        post_burnin_accepted = np.sum(accepted_mask)
        post_burnin_rate = (post_burnin_accepted / post_burnin_total) * 100 if post_burnin_total > 0 else 0
        
        print(f"Post burn-in samples: {post_burnin_total}")
        print(f"Post burn-in accepted: {post_burnin_accepted} ({post_burnin_rate:.1f}%)")
        
        # Create DataFrame from accepted samples only
        df = pd.DataFrame(data=params_accepted.T, columns=res.parnames)
        
        # Create accepted statistics array for trace plot (burn-in + post-burn-in accepted)
        # Filter accepted samples from the full stats array (including burn-in)
        full_accepted_mask = accept.astype(bool)
        stats_all_accepted = stats[full_accepted_mask]
        
        # Split accepted stats into burn-in and post-burn-in portions
        # Count accepted samples up to burn_in point
        accepted_burnin_count = np.sum(accept[:burn_in])
        stats_accepted_burnin = stats_all_accepted[:accepted_burnin_count] if accepted_burnin_count > 0 else np.array([])
        stats_accepted_postburnin = stats_all_accepted[accepted_burnin_count:] if accepted_burnin_count < len(stats_all_accepted) else np.array([])
        
        # Create output folder based on galaxy name (without .fits)
        folder_name = galaxy_name.replace('.fits', '')
        os.makedirs(folder_name, exist_ok=True)
        
        # Save chain in the folder with model type
        if model_type:
            model_suffix = f"_{model_type.replace(' ', '_')}"
        else:
            model_suffix = ""
        chain_filename = os.path.join(folder_name, f"{galaxy_name}{model_suffix}_chain.csv")
        df.to_csv(chain_filename, index=False)
        
        # Return the full stats array including all iterations (not just accepted)
        if return_stats:
            return df, stats, accept
        else:
            return df


class BetaFittingTool:
    """Main class orchestrating the beta fitting workflow."""
    
    def __init__(self, data_directory: str = "."):
        self.data_directory = Path(data_directory)
        self.config = FittingConfig()
        self.mcmc_analyzer = MCMCAnalyzer(self.config)
        
        self.current_galaxy = None
        self.current_model = None
        self.current_scale = "original"
        
        self._discover_galaxies()
    
    def _discover_galaxies(self):
        """Find all FITS files in the data directory."""
        self.galaxy_files = list(self.data_directory.glob("*.fits"))
        if not self.galaxy_files:
            raise FileNotFoundError(f"No FITS files found in {self.data_directory}")
    
    def load_galaxy(self, galaxy_path: Union[str, Path], scale: str = "original"):
        """Load a galaxy image and setup for fitting."""
        self.current_galaxy = GalaxyImage(galaxy_path)
        self.current_scale = scale
        
        # Load image in Sherpa
        reset()
        clean()
        load_image(str(galaxy_path))
        
        # Apply scale if not original
        if scale != "original":
            scale_val = int(scale.split()[0].split("x")[0])
            center = self.current_galaxy.center
            if scale_val > center[0] * 2:
                raise ValueError("Invalid scale. Original image is smaller.")
            notice2d(f"BOX({center[0]+0.5},{center[1]+0.5},{scale_val},{scale_val})")
        
        set_coord("image")
        print(f"Loaded galaxy: {self.current_galaxy.name}")
        print(f"Image shape: {self.current_galaxy.shape}")
        print(f"Size: {scale}")
    
    def setup_model(self, model_type: str):
        """Setup the fitting model."""
        if self.current_galaxy is None:
            raise ValueError("No galaxy loaded. Call load_galaxy() first.")
        
        self.current_model = BetaModel(model_type, self.current_galaxy.center)
        
        # Reload the data since BetaModel.__init__ calls reset() and clean()
        load_image(str(self.current_galaxy.filepath))
        
        # Apply scale if not original
        if self.current_scale != "original":
            scale_val = int(self.current_scale.split()[0].split("x")[0])
            center = self.current_galaxy.center
            if scale_val > center[0] * 2:
                raise ValueError("Invalid scale. Original image is smaller.")
            notice2d(f"BOX({center[0]+0.5},{center[1]+0.5},{scale_val},{scale_val})")
        
        set_coord("image")
        
        # Initial parameter guessing disabled to preserve manually set xpos/ypos
        # The guess() function would override all parameters including positions
        # Manual parameter setup in _setup_initial_parameters() is used instead
        
        # Set default frozen parameters
        self.set_default_frozen_parameters()
        
        print(f"Setup model: {model_type}")
    
    def fit_model(self, method: str = None, statistic: str = None):
        """Fit the current model to the data."""
        if self.current_model is None:
            raise ValueError("No model setup. Call setup_model() first.")
        
        # Set fitting parameters
        if method:
            set_method(method)
        if statistic:
            set_stat(statistic)
        
        print("Fitting model...\n")
        fit()
        print("\nFit completed!")
        
        # Show fit results
        self.show_fit_results()
    
    def show_fit_results(self):
        """Display fit results."""
        if self.current_model is None:
            raise ValueError("No model fitted.")
        
        print("\nFit Results:")
        print("=" * 50)
        print(get_fit_results())
    
    def update_parameter(self, param_name: str, value: float):
        """Update a model parameter and refresh the model."""
        if self.current_model is None:
            raise ValueError("No model loaded.")
        
        from sherpa.astro.ui import set_par, get_model_component
        
        # Parse parameter name (e.g., 'b1.r0' -> component='b1', param='r0')
        component_name, param = param_name.split('.')
        component = get_model_component(component_name)
        
        # Update the parameter
        setattr(getattr(component, param), 'val', value)
    
    def freeze_parameter(self, param_name: str, freeze_param: bool = True):
        """Freeze or unfreeze a model parameter."""
        if self.current_model is None:
            raise ValueError("No model loaded.")
        
        from sherpa.astro.ui import freeze, thaw, get_model_component
        
        # Parse parameter name (e.g., 'b1.r0' -> component='b1', param='r0')
        component_name, param = param_name.split('.')
        component = get_model_component(component_name)
        
        # Freeze or unfreeze the parameter
        if freeze_param:
            freeze(getattr(component, param))
        else:
            thaw(getattr(component, param))
    
    
    def set_default_frozen_parameters(self):
        """Set default frozen parameters according to specification."""
        if self.current_model is None:
            return
        
        # Parameters to freeze by default: B1 X, B1 Y, B1 Alpha, B2 X, B2 Y, B2 Alpha, etc.
        default_frozen = ['b1.xpos', 'b1.ypos', 'b1.alpha', 'b2.xpos', 'b2.ypos', 'b2.alpha',
                         'b3.xpos', 'b3.ypos', 'b3.alpha', 'b4.xpos', 'b4.ypos', 'b4.alpha',
                         'b5.xpos', 'b5.ypos', 'b5.alpha']
        
        for param_name in default_frozen:
            try:
                self.freeze_parameter(param_name, freeze_param=True)
            except:
                # Parameter might not exist for this model type
                pass
    
    def get_parameter_info(self):
        """Get current parameter values and ranges for the active model."""
        if self.current_model is None:
            return {}
        
        from sherpa.astro.ui import get_model_component
        
        params = {}
        model_type = self.current_model.model_type
        
        def _get_param_info(component, param_name):
            """Helper function to extract parameter information."""
            param_obj = getattr(component, param_name)
            
            return {
                'val': param_obj.val, 
                'min': param_obj.min, 
                'max': param_obj.max, 
                'name': f'{component.name}.{param_name}', 
                'frozen': param_obj.frozen
            }
        
        # Primary beta component
        try:
            b1 = get_model_component('b1')
            params['b1.r0'] = _get_param_info(b1, 'r0')
            params['b1.ampl'] = _get_param_info(b1, 'ampl')
            params['b1.alpha'] = _get_param_info(b1, 'alpha')
            params['b1.xpos'] = _get_param_info(b1, 'xpos')
            params['b1.ypos'] = _get_param_info(b1, 'ypos')
        except:
            pass
        
        # Secondary beta component (for double-beta, triple-beta, four-beta and five-beta models)
        if "Double" in model_type or "Triple" in model_type or "Four" in model_type or "Five" in model_type:
            try:
                b2 = get_model_component('b2')
                params['b2.r0'] = _get_param_info(b2, 'r0')
                params['b2.ampl'] = _get_param_info(b2, 'ampl')
                params['b2.alpha'] = _get_param_info(b2, 'alpha')
                params['b2.xpos'] = _get_param_info(b2, 'xpos')
                params['b2.ypos'] = _get_param_info(b2, 'ypos')
            except:
                pass
        
        # Tertiary beta component (for triple-beta, four-beta and five-beta models)
        if "Triple" in model_type or "Four" in model_type or "Five" in model_type:
            try:
                b3 = get_model_component('b3')
                params['b3.r0'] = _get_param_info(b3, 'r0')
                params['b3.ampl'] = _get_param_info(b3, 'ampl')
                params['b3.alpha'] = _get_param_info(b3, 'alpha')
                params['b3.xpos'] = _get_param_info(b3, 'xpos')
                params['b3.ypos'] = _get_param_info(b3, 'ypos')
            except:
                pass
        
        # Quaternary beta component (for four-beta and five-beta models)
        if "Four" in model_type or "Five" in model_type:
            try:
                b4 = get_model_component('b4')
                params['b4.r0'] = _get_param_info(b4, 'r0')
                params['b4.ampl'] = _get_param_info(b4, 'ampl')
                params['b4.alpha'] = _get_param_info(b4, 'alpha')
                params['b4.xpos'] = _get_param_info(b4, 'xpos')
                params['b4.ypos'] = _get_param_info(b4, 'ypos')
            except:
                pass
        
        # Quinary beta component (for five-beta models)
        if "Five" in model_type:
            try:
                b5 = get_model_component('b5')
                params['b5.r0'] = _get_param_info(b5, 'r0')
                params['b5.ampl'] = _get_param_info(b5, 'ampl')
                params['b5.alpha'] = _get_param_info(b5, 'alpha')
                params['b5.xpos'] = _get_param_info(b5, 'xpos')
                params['b5.ypos'] = _get_param_info(b5, 'ypos')
            except:
                pass
        
        # Gaussian component (for models with gauss)
        if "gauss" in model_type:
            try:
                g1 = get_model_component('g1')
                params['g1.fwhm'] = _get_param_info(g1, 'fwhm')
                params['g1.ampl'] = _get_param_info(g1, 'ampl')
                params['g1.xpos'] = _get_param_info(g1, 'xpos')
                params['g1.ypos'] = _get_param_info(g1, 'ypos')
            except:
                pass
        
        # Background
        try:
            bkg = get_model_component('bkg')
            params['bkg.c0'] = _get_param_info(bkg, 'c0')
        except:
            pass
        
        return params
    
    def get_radial_profile_data(self) -> Optional[RadialProfileData]:
        """Extract radial profile data for plotting."""
        try:
            # Get profile data with fallback for multi-component models
            from sherpa_contrib.profiles import get_data_prof, get_model_prof
            prof = get_data_prof(model=b1)
            rad_pixels = (prof.xlo + prof.xhi) / 2
            y, yerr = prof.y, prof.yerr
            y_fit = get_model_prof(model=b1).y
            
            # Get pixel scale from WCS if available
            pixel_scale = calculate_pixel_scale_from_wcs(
                self.current_galaxy.wcs if self.current_galaxy else None, 
                self.config.pixel_scale
            )
            
            # Convert radius from pixels to arcseconds
            rad_arcsec = rad_pixels * pixel_scale
            
            # Get individual components
            components = {}
            param_info = self._get_current_parameter_values()
            
            # Calculate individual beta components (using pixel radius as primary)
            if 'b1' in param_info:
                components['b1'] = self._calculate_beta1d_profile(rad_pixels, param_info['b1'], pixel_scale=None)
            if 'b2' in param_info:
                components['b2'] = self._calculate_beta1d_profile(rad_pixels, param_info['b2'], pixel_scale=None)
            if 'b3' in param_info:
                components['b3'] = self._calculate_beta1d_profile(rad_pixels, param_info['b3'], pixel_scale=None)
            if 'b4' in param_info:
                components['b4'] = self._calculate_beta1d_profile(rad_pixels, param_info['b4'], pixel_scale=None)
            if 'b5' in param_info:
                components['b5'] = self._calculate_beta1d_profile(rad_pixels, param_info['b5'], pixel_scale=None)
            if 'bkg' in param_info:
                components['bkg'] = np.full_like(rad_pixels, param_info['bkg']['c0'])
            
            return RadialProfileData(
                radius_pixels=rad_pixels,
                radius_arcsec=rad_arcsec,
                data=y,
                data_error=yerr,
                model=y_fit,
                components=components,
                pixel_scale=pixel_scale
            )
            
        except Exception as e:
            print(f"Error extracting radial profile data: {e}")
            return None
    
    def get_image_data(self) -> Optional[ImageData]:
        """Extract image data for plotting."""
        try:
            from sherpa.astro.ui import get_data, get_model_image, get_resid_image
            
            # Get size information
            if self.current_scale == "original":
                size = self.current_galaxy.center[0]
            else:
                size = float(self.current_scale.split()[0].split("x")[0]) / 2
            
            # Get images
            data = self.current_galaxy.data
            model_data = get_model_image().y.reshape(get_data().shape)
            # Calculate residuals using (data - model) / model**0.5 instead of Sherpa's default
            residual_data = (data - model_data) / np.sqrt(model_data)
            
            # Get smoothing parameter
            smoothing_std = self.config.smoothing_map.get(size * 2 // 64 * 64, 1)
            
            return ImageData(
                data=data,
                model=model_data,
                residual=residual_data,
                center=self.current_galaxy.center,
                size=size,
                smoothing_std=smoothing_std,
                wcs=self.current_galaxy.wcs if self.current_galaxy else None
            )
            
        except Exception as e:
            print(f"Error extracting image data: {e}")
            return None
    def get_beta_model_params(self) -> Dict[str, Dict[str, float]]:
        """Extract beta model parameters for plotting overlay."""
        beta_params = {}
        
        if self.current_model is None:
            return beta_params
        
        try:
            from sherpa.astro.ui import get_par
            
            # Primary beta component
            try:
                beta_params['b1'] = {
                    'xpos': get_par('b1.xpos').val,
                    'ypos': get_par('b1.ypos').val,
                    'r0': get_par('b1.r0').val,
                    'ampl': get_par('b1.ampl').val,
                    'alpha': get_par('b1.alpha').val
                }
            except Exception:
                pass
            
            # Secondary beta component
            if "Double" in self.current_model.model_type or "Triple" in self.current_model.model_type or "Four" in self.current_model.model_type or "Five" in self.current_model.model_type:
                try:
                    beta_params['b2'] = {
                        'xpos': get_par('b2.xpos').val,
                        'ypos': get_par('b2.ypos').val,
                        'r0': get_par('b2.r0').val,
                        'ampl': get_par('b2.ampl').val,
                        'alpha': get_par('b2.alpha').val
                    }
                except Exception:
                    pass
            
            # Tertiary beta component
            if "Triple" in self.current_model.model_type or "Four" in self.current_model.model_type or "Five" in self.current_model.model_type:
                try:
                    beta_params['b3'] = {
                        'xpos': get_par('b3.xpos').val,
                        'ypos': get_par('b3.ypos').val,
                        'r0': get_par('b3.r0').val,
                        'ampl': get_par('b3.ampl').val,
                        'alpha': get_par('b3.alpha').val
                    }
                except Exception:
                    pass
            
            # Quaternary beta component
            if "Four" in self.current_model.model_type or "Five" in self.current_model.model_type:
                try:
                    beta_params['b4'] = {
                        'xpos': get_par('b4.xpos').val,
                        'ypos': get_par('b4.ypos').val,
                        'r0': get_par('b4.r0').val,
                        'ampl': get_par('b4.ampl').val,
                        'alpha': get_par('b4.alpha').val
                    }
                except Exception:
                    pass
            
            # Quinary beta component
            if "Five" in self.current_model.model_type:
                try:
                    beta_params['b5'] = {
                        'xpos': get_par('b5.xpos').val,
                        'ypos': get_par('b5.ypos').val,
                        'r0': get_par('b5.r0').val,
                        'ampl': get_par('b5.ampl').val,
                        'alpha': get_par('b5.alpha').val
                    }
                except Exception:
                    pass
            
        except Exception as e:
            print(f"Warning: Could not extract beta model parameters: {e}")
        
        return beta_params
    
    def _get_current_parameter_values(self):
        """Extract current parameter values from Sherpa model."""
        param_values = {}
        
        try:
            from sherpa.astro.ui import get_par
            
            # Primary beta component
            try:
                param_values['b1'] = {
                    'ampl': get_par('b1.ampl').val,
                    'r0': get_par('b1.r0').val,
                    'alpha': get_par('b1.alpha').val,
                    'xpos': get_par('b1.xpos').val,
                    'ypos': get_par('b1.ypos').val
                }
            except Exception:
                pass
            
            # Secondary beta component
            try:
                param_values['b2'] = {
                    'ampl': get_par('b2.ampl').val,
                    'r0': get_par('b2.r0').val,
                    'alpha': get_par('b2.alpha').val,
                    'xpos': get_par('b2.xpos').val,
                    'ypos': get_par('b2.ypos').val
                }
            except Exception:
                pass
        
            # Tertiary beta component
            try:
                param_values['b3'] = {
                    'ampl': get_par('b3.ampl').val,
                    'r0': get_par('b3.r0').val,
                    'alpha': get_par('b3.alpha').val,
                    'xpos': get_par('b3.xpos').val,
                    'ypos': get_par('b3.ypos').val
                }
            except Exception:
                pass
            
            # Quaternary beta component
            try:
                param_values['b4'] = {
                    'ampl': get_par('b4.ampl').val,
                    'r0': get_par('b4.r0').val,
                    'alpha': get_par('b4.alpha').val,
                    'xpos': get_par('b4.xpos').val,
                    'ypos': get_par('b4.ypos').val
                }
            except Exception:
                pass
            
            # Quinary beta component
            try:
                param_values['b5'] = {
                    'ampl': get_par('b5.ampl').val,
                    'r0': get_par('b5.r0').val,
                    'alpha': get_par('b5.alpha').val,
                    'xpos': get_par('b5.xpos').val,
                    'ypos': get_par('b5.ypos').val
                }
            except Exception:
                pass
            
            # Background component
            try:
                param_values['bkg'] = {
                    'c0': get_par('bkg.c0').val
                }
            except Exception:
                pass
            
            # Gaussian component
            try:
                param_values['g1'] = {
                    'ampl': get_par('g1.ampl').val,
                    'fwhm': get_par('g1.fwhm').val,
                    'xpos': get_par('g1.xpos').val,
                    'ypos': get_par('g1.ypos').val
                }
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: Could not extract parameter values: {e}")
        
        return param_values
    
    
    def _calculate_beta1d_profile(self, radius: np.ndarray, beta_params: dict, pixel_scale: float = None) -> np.ndarray:
        """Calculate 1D beta profile from parameter dictionary."""
        try:
            ampl = beta_params['ampl']
            r0_pixels = beta_params['r0']  # r0 is in pixels
            alpha = beta_params['alpha']
            
            # Convert r0 from pixels to arcseconds if pixel_scale is provided
            if pixel_scale is not None:
                r0_arcsec = r0_pixels * pixel_scale
                # radius is now in arcseconds, r0_arcsec is also in arcseconds
                return ampl * (1 + (radius / r0_arcsec) ** 2) ** (-alpha)
            else:
                # Fallback to pixel-based calculation
                return ampl * (1 + (radius / r0_pixels) ** 2) ** (-alpha)
        except Exception as e:
            print(f"Error calculating beta profile: {e}")
            return np.zeros_like(radius)
    
    def run_mcmc_analysis(self, length: int = None, burn_in: int = None) -> pd.DataFrame:
        """Run MCMC analysis on the current model."""
        if self.current_model is None:
            raise ValueError("No model fitted.")
        
        length = length or self.config.default_mcmc_length
        burn_in = burn_in or self.config.default_burn_in
        
        return self.mcmc_analyzer.run_mcmc(length, burn_in, self.current_galaxy.name)
    
    def save_model(self, output_dir: str = "Beta_models/Prefits"):
        """Save the current model to a text file."""
        if self.current_model is None:
            raise ValueError("No model to save.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / f"{self.current_galaxy.name}.txt"
        with open(filename, "w") as f:
            print(get_model(), file=f)
        
        print(f"Model saved to {filename}")
    
    def save_residual(self, output_dir: str = "Beta_models/Residuals"):
        """Save the residual image."""
        if self.current_model is None:
            raise ValueError("No model fitted.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get residual data
        resid = get_resid_image().y.reshape(get_data().shape)
        
        # Create CCDData with WCS
        ccd = CCDData(resid, unit="adu", wcs=self.current_galaxy.wcs)
        
        filename = output_path / f"{self.current_galaxy.name}_resid.fits"
        ccd.write(filename, overwrite=True)
        
        print(f"Residual saved to {filename}")
