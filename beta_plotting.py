#!/usr/bin/env python3
"""
Beta Profile Plotting Module
============================

This module contains purely matplotlib-based plotting functions that consume
data structures from the beta_modeling module. It has no dependencies on Sherpa.

Author: Extracted from beta_fitting_improved.py
"""

import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Ellipse, Circle
from matplotlib import gridspec
from matplotlib import patheffects
from scipy.stats import gaussian_kde

# Import utility functions
from utils import suppress_astropy_warnings
suppress_astropy_warnings()

from astropy.convolution import Gaussian2DKernel, convolve

# Import data classes from modeling module
from beta_modeling import RadialProfileData, ImageData, FittingConfig


class PlottingManager:
    """Handle all plotting operations using only matplotlib."""
    
    def __init__(self, config: FittingConfig):
        self.config = config
    
    def _bin_image_for_display(self, image: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, int]:
        """Bin image to target size for display purposes only.
        
        Args:
            image: Input image array
            target_size: Target size for the largest dimension
            
        Returns:
            Tuple of (binned image array, binning factor)
        """
        if image is None:
            return None, 1
            
        height, width = image.shape
        max_dim = max(height, width)
        
        # If image is already small enough, return as-is
        if max_dim <= target_size:
            return image, 1
            
        # Calculate binning factor
        bin_factor = int(np.ceil(max_dim / target_size))
        
        # Calculate new dimensions after binning
        new_height = height // bin_factor
        new_width = width // bin_factor
        
        # Trim image to be divisible by bin_factor
        trimmed_height = new_height * bin_factor
        trimmed_width = new_width * bin_factor
        
        trimmed_image = image[:trimmed_height, :trimmed_width]
        
        # Reshape and bin the image
        binned = trimmed_image.reshape(new_height, bin_factor, new_width, bin_factor)
        binned = binned.mean(axis=(1, 3))
        
        return binned, bin_factor
    
    def create_comprehensive_plot(self, profile_data: Optional[RadialProfileData], 
                                image_data: Optional[ImageData], galaxy_name: str,
                                model_type: str = "", fitted: bool = False,
                                beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                                distance_scale: Optional[float] = None) -> plt.Figure:
        """Create comprehensive plot with data, model, residual, and radial profile."""
        fig = plt.figure(figsize=(11, 11), dpi=100)
        
        # Create layout: combined radial plot, original image, model image, residual image
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.2, wspace=0.2)
        
        # Combined radial plot (radial + residual)
        gs_radial = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0, 0], height_ratios=[4, 1], hspace=0.05)
        ax_radial = fig.add_subplot(gs_radial[0])     # Main radial plot
        ax_residual_plot = fig.add_subplot(gs_radial[1])  # Residual subplot
        
        # Image plots
        ax_data = fig.add_subplot(gs[1, 0])          # Original image
        ax_model = fig.add_subplot(gs[1, 1])         # Model image
        ax_residual_img = fig.add_subplot(gs[0, 1])  # Residual image

        # Set consistent font sizes
        title_fontsize = 12
        label_fontsize = 10
        
        # Plot data image if available
        if image_data is not None:
            self._plot_data_image(ax_data, image_data, galaxy_name, beta_params, distance_scale)
            ax_data.set_title("Data", fontsize=title_fontsize)
            
            # Plot model and residual images
            self._plot_model_image(ax_model, image_data, beta_params, distance_scale)
            ax_model.set_title("Model", fontsize=title_fontsize)
            
            self._plot_residual_image(ax_residual_img, image_data, beta_params, distance_scale)
            ax_residual_img.set_title("Residual", fontsize=title_fontsize)
        else:
            # No image data available
            for ax, title in zip([ax_data, ax_model, ax_residual_img], 
                               ["Data", "Model", "Residual"]):
                ax.text(0.5, 0.5, 'No image\\ndata available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=label_fontsize)
                ax.set_title(title, fontsize=title_fontsize)
        
        # Plot radial profile if available
        if profile_data is not None:
            self._plot_radial_profile_with_residual(ax_radial, ax_residual_plot, 
                                                   profile_data, fitted, beta_params, distance_scale,
                                                   label_fontsize=label_fontsize)
        else:
            # No profile data
            ax_radial.text(0.5, 0.5, 'No radial\\nprofile available', 
                          ha='center', va='center', transform=ax_radial.transAxes, fontsize=label_fontsize)
            ax_residual_plot.text(0.5, 0.5, 'No data', 
                                 ha='center', va='center', transform=ax_residual_plot.transAxes, fontsize=label_fontsize)
        
        return fig
    
    def create_data_plot(self, image_data: ImageData, galaxy_name: str, 
                        beta_params: Optional[Dict[str, Dict[str, float]]] = None) -> plt.Figure:
        """Create simple data-only plot."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        self._plot_data_image(ax, image_data, galaxy_name, beta_params)
        ax.set_title(f"Galaxy: {galaxy_name}")
        
        return fig
    
    def _plot_radial_profile_with_residual(self, ax_main, ax_res, 
                                         profile_data: RadialProfileData, 
                                         fitted: bool = False,
                                         beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                                         distance_scale: Optional[float] = None,
                                         label_fontsize: int = 10):
        """Plot radial profile with residual subplot below, including individual components."""
        # Conversion factor
        c = profile_data.pixel_scale ** 2
        
        # Plot data and total fit in main subplot (using pixels for x-axis)
        good_data = profile_data.data_error / profile_data.data < 2
        ax_main.errorbar(profile_data.radius_pixels[good_data], 
                        profile_data.data[good_data] / c, 
                        yerr=profile_data.data_error[good_data] / c, 
                        fmt='o', markersize=2, capsize=1, label='Data', alpha=0.7, zorder=1)
        ax_main.plot(profile_data.radius_pixels, profile_data.model / c, 'r-', 
                    linewidth=2, label='Total Model', zorder=2)
        
        # Plot individual components  
        # Get beta parameters from the plotting manager call to show r0 values
        # This should be updated to pass beta_params to this method
        
        # Plot individual components with standard colors and styles
        colors = ['blue', 'green', 'purple', 'brown', 'pink']
        linestyles = ['--', ':', '-.', '-', '--']
        
        beta_comp_idx = 0
        for comp_name, comp_data in profile_data.components.items():
            if comp_name.startswith('b') and comp_name != 'bkg':
                color = colors[beta_comp_idx % len(colors)]
                linestyle = linestyles[beta_comp_idx % len(linestyles)]
                
                # Create label with indexed r0 notation and r0 value
                label = f'Beta {beta_comp_idx + 1}'
                
                # Add r0 value to label if beta_params are available
                if beta_params and comp_name in beta_params:
                    r0_pixels = beta_params[comp_name]['r0']
                    # Convert to appropriate units
                    if distance_scale is not None and distance_scale > 0:
                        # Convert pixels -> arcsec -> parsec -> kpc
                        r0_arcsec = r0_pixels * profile_data.pixel_scale
                        r0_kpc = (r0_arcsec * distance_scale) / 1000.0
                        label += f' ($r_{{0}}={r0_kpc:.2f}$ kpc)'
                    else:
                        # Convert pixels -> arcsec
                        r0_arcsec = r0_pixels * profile_data.pixel_scale
                        label += f' ($r_{{0}}={r0_arcsec:.1f}$")'
                
                ax_main.plot(profile_data.radius_pixels, comp_data / c, linestyle, 
                           color=color, linewidth=1.5, 
                           label=label, alpha=0.8, zorder=2)
                beta_comp_idx += 1
            elif comp_name == 'bkg':
                ax_main.axhline(comp_data[0] / c, color='gray', linestyle='-', 
                              linewidth=1, alpha=0.7, 
                              label=f'Background ({comp_data[0]:.3f})')
        
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_ylabel("Counts/pixel", fontsize=label_fontsize)
        ax_main.legend(fontsize=label_fontsize - 1, loc='lower left')
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelsize=label_fontsize)
        ax_main.set_xlabel("Radius (pixels)", fontsize=label_fontsize)
        
        # Create twin x-axis for arcseconds
        ax_top = ax_main.twiny()
        ax_top.set_xscale("log")
        
        # Function to convert pixel to arcsec for twin axis
        def pixel_to_arcsec(x):
            return x * profile_data.pixel_scale
        
        def arcsec_to_pixel(x):
            return x / profile_data.pixel_scale
        
        # Get pixel axis limits
        pixel_min, pixel_max = ax_main.get_xlim()
        
        # Set twin axis to show arcsec values corresponding to pixel values
        ax_top.set_xlim(pixel_to_arcsec(pixel_min), pixel_to_arcsec(pixel_max))
        
        # Set xlabel for twin axis
        ax_top.set_xlabel("Radius (arcsec)", fontsize=label_fontsize)
        ax_top.tick_params(labelsize=label_fontsize)

        # Plot residuals in bottom subplot (using pixels)
        residuals = profile_data.get_residuals()
        ax_res.errorbar(profile_data.radius_pixels[good_data], residuals[good_data], 
                       yerr=1, fmt='o', markersize=2, capsize=1, alpha=0.7)
        ax_res.axhline(0, ls='--', color='black', alpha=0.5)
        ax_res.set_ylabel("Residual (σ)", fontsize=label_fontsize)
        
        ax_res.set_xscale("log")
        ax_res.set_xlabel("Radius (pixels)", fontsize=label_fontsize)
        ax_res.tick_params(labelsize=label_fontsize)
        ax_res.grid(True, alpha=0.3)
    
    def _plot_data_image(self, ax, image_data: ImageData, galaxy_name: str, 
                        beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                        distance_scale: Optional[float] = None):
        """Plot the data image with center and parameter visualization."""
        # Extract cutout
        data_cutout = image_data.data_cutout
        
        # Apply smoothing
        if image_data.smoothing_std > 0:
            data_cutout = convolve(data_cutout, boundary="extend", nan_treatment="fill",
                                  fill_value=np.amin(data_cutout),
                                  kernel=Gaussian2DKernel(x_stddev=image_data.smoothing_std, 
                                                        y_stddev=image_data.smoothing_std))
        
        # Bin for display if image is large
        data_cutout_display, bin_factor = self._bin_image_for_display(data_cutout, target_size=512)
        
        # Plot with log scale
        ax.imshow(np.log10(data_cutout_display + 1), origin="lower")
        
        # Add basic annotations
        self._add_image_annotations(ax, image_data.size, bin_factor)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale, bin_factor)
        
        # Add angular scale bar
        self._add_angular_scale_bar(ax, image_data, distance_scale=distance_scale)
    
    def _plot_model_image(self, ax, image_data: ImageData, 
                         beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                         distance_scale: Optional[float] = None):
        """Plot the model image."""
        model_cutout = image_data.model_cutout
        
        # Bin for display if image is large
        model_cutout_display, bin_factor = self._bin_image_for_display(model_cutout, target_size=512)
        
        cmap = plt.cm.viridis
        ax.imshow(model_cutout_display + 1, origin="lower", norm=LogNorm(), cmap=cmap)
        self._add_image_annotations(ax, image_data.size, bin_factor)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale, bin_factor)
        
        # Add angular scale bar
        self._add_angular_scale_bar(ax, image_data, distance_scale=distance_scale)
    
    def _plot_residual_image(self, ax, image_data: ImageData, 
                            beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                            distance_scale: Optional[float] = None):
        """Plot the residual image."""
        residual_cutout = image_data.residual_cutout
        
        # Apply smoothing if specified
        if image_data.smoothing_std > 0:
            residual_cutout = convolve(residual_cutout, boundary="extend", 
                                     nan_treatment="fill",
                                     fill_value=np.amin(residual_cutout),
                                     kernel=Gaussian2DKernel(x_stddev=image_data.smoothing_std, 
                                                           y_stddev=image_data.smoothing_std))
        
        # Clip extreme values
        array = residual_cutout.flatten()
        std = np.std(array) * 4
        residual_cutout = np.clip(residual_cutout, -std, std)
        
        # Bin for display if image is large
        residual_cutout_display, bin_factor = self._bin_image_for_display(residual_cutout, target_size=512)
        
        ax.imshow(residual_cutout_display, origin="lower", cmap="coolwarm")
        self._add_image_annotations(ax, image_data.size, bin_factor)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale, bin_factor)
        
        # Add angular scale bar
        self._add_angular_scale_bar(ax, image_data, distance_scale=distance_scale)
    
    def _add_image_annotations(self, ax, size: float, bin_factor: int = 1):
        """Add common annotations to image plots."""
        # Calculate ticks in original coordinates, rounded to nice values
        max_tick = int(2*size // 64 * 64)
        # Ensure max tick doesn't go below the cutout size when it's close to a multiple of 64
        if max_tick < 2*size - 16:  # If rounding down loses more than 16 pixels
            max_tick = int(2*size // 64 * 64 + 64)  # Round up to next multiple
        
        # Create evenly spaced ticks from 0 to max_tick, ensuring the endpoint is included
        original_ticks = np.linspace(0, max_tick, 5)
        # Convert to binned coordinates for display
        binned_ticks = original_ticks / bin_factor
        
        ax.set_xticks(binned_ticks)
        ax.set_yticks(binned_ticks)
        # Set labels to show original coordinates
        ax.set_xticklabels([f'{int(tick)}' for tick in original_ticks])
        ax.set_yticklabels([f'{int(tick)}' for tick in original_ticks])
    
    
    def _add_angular_scale_bar(self, ax, image_data: ImageData, scale_arcsec: float = 20.0, distance_scale: Optional[float] = None):
        """Add angular scale bar to the bottom left of the image."""
        from matplotlib import patheffects
        
        # Get pixel scale
        pixel_scale = image_data.get_pixel_scale_arcsec() if hasattr(image_data, 'get_pixel_scale_arcsec') else 0.492
        
        # Convert scale from arcseconds to pixels in the original image coordinates
        scale_pixels_original = scale_arcsec / pixel_scale
        
        # Get current axis limits to determine the displayed image size
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        displayed_width = xlim[1] - xlim[0]
        displayed_height = ylim[1] - ylim[0]
        
        # Calculate scale bar length as a fraction of the displayed image width
        scale_bar_fraction = scale_pixels_original / (2 * image_data.size)
        
        # Position scale bar using axes coordinates (0-1 range, independent of binning)
        margin_x_frac = 0.12  # 12% from left edge
        margin_y_frac = 0.12  # 12% from bottom edge
        
        # Scale bar coordinates in axes fraction (0-1)
        start_x_frac = margin_x_frac
        start_y_frac = margin_y_frac
        end_x_frac = start_x_frac + scale_bar_fraction
        end_y_frac = start_y_frac
        
        # Draw black outline line (slightly thicker and longer for contrast)
        ax.plot([start_x_frac-0.005, end_x_frac+0.005], [start_y_frac, start_y_frac], 
               'k-', linewidth=5, solid_capstyle='butt', transform=ax.transAxes)
        # Draw white scale bar line on top
        ax.plot([start_x_frac, end_x_frac], [start_y_frac, start_y_frac], 
               'w-', linewidth=3, solid_capstyle='butt', transform=ax.transAxes)
        
        # Add text label for arcseconds
        text_x_frac = start_x_frac + scale_bar_fraction / 2  # Center the text over the line
        text_y_frac = start_y_frac + 0.03  # Slightly above the line
        
        # Add arcsec text with black outline
        ax.text(text_x_frac, text_y_frac, f'{scale_arcsec:.0f} arcsec', 
               color='white', fontweight='bold', fontsize=10, 
               ha='center', va='bottom', transform=ax.transAxes,
               path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        
        # Add kpc scale if distance scale is provided
        if distance_scale is not None and distance_scale > 0:
            # Convert arcseconds to kpc: arcsec * pc/arcsec / 1000
            scale_kpc = scale_arcsec * distance_scale / 1000.0
            kpc_text_y_frac = start_y_frac - 0.03  # Below the line
            
            # Add kpc text with black outline
            ax.text(text_x_frac, kpc_text_y_frac, f'{scale_kpc:.1f} kpc', 
                   color='white', fontweight='bold', fontsize=10, 
                   ha='center', va='top', transform=ax.transAxes,
                   path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    def _add_beta_model_overlays(self, ax, image_data: ImageData, 
                                beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                                distance_scale: Optional[float] = None,
                                bin_factor: int = 1):
        """Add beta model center markers and core radius circles to the image."""
        if not beta_params:
            return
        
        # Get cutout center coordinates
        cutout_center_x = image_data.size
        cutout_center_y = image_data.size
        
        # Original image center
        image_center_x, image_center_y = image_data.center
        
        # Colors and styles for different beta components
        colors = ['yellow', 'orange', 'purple', 'brown', 'pink']
        linestyles = ['-', '--', '-.', '-', '--']
        
        for i, (comp_name, color, linestyle) in enumerate(zip(['b1', 'b2', 'b3', 'b4', 'b5'], colors, linestyles)):
            if comp_name in beta_params:
                params = beta_params[comp_name]
                
                # Get model center position (relative to image)
                model_x = params['xpos']
                model_y = params['ypos']
                core_radius = params['r0']
                
                # Convert to cutout coordinates
                cutout_x = model_x - (image_center_x - image_data.size)
                cutout_y = model_y - (image_center_y - image_data.size)
                
                # Adjust for binning factor
                cutout_x_binned = cutout_x / bin_factor
                cutout_y_binned = cutout_y / bin_factor
                core_radius_binned = core_radius / bin_factor
                
                # Add center marker (smaller)
                ax.plot(cutout_x_binned, cutout_y_binned, '+', color=color, markersize=6, 
                       markeredgewidth=1.5, label=f'{comp_name.upper()} center')
                
                # Add core radius circle
                if core_radius > 0:
                    circle = Circle((cutout_x_binned, cutout_y_binned), core_radius_binned, 
                                  fill=False, color=color, linewidth=2, 
                                  linestyle=linestyle, alpha=0.8)
                    ax.add_patch(circle)
                    
                    # Add text label for core radius
                    pixel_scale = image_data.get_pixel_scale_arcsec() if hasattr(image_data, 'get_pixel_scale_arcsec') else 0.492
                    text_x = cutout_x_binned + core_radius_binned * 1.0
                    text_y = cutout_y_binned + core_radius_binned * 1.0
                    
                    if distance_scale is not None and distance_scale > 0:
                        # Convert to kpc: pixels -> arcsec -> parsec -> kpc
                        # distance_scale is now pc/arcsec, so multiply instead of divide
                        core_radius_arcsec = core_radius * pixel_scale
                        core_radius_parsec = core_radius_arcsec * distance_scale
                        core_radius_kpc = core_radius_parsec / 1000.0
                        label_text = f'$r_{{0,{i+1}}}={core_radius_kpc:.2f}$ kpc'
                    else:
                        # Use arcseconds as before
                        core_radius_arcsec = core_radius * pixel_scale
                        label_text = f'$r_{{0,{i+1}}}={core_radius_arcsec:.1f}$"'
                    
                    # Use white text for data and model, black for residual
                    text_color = 'white' if ax.get_title() in ['Data', 'Model'] else 'black'
                    ax.text(
                        text_x, text_y, label_text,
                        color='white', fontweight='normal', fontsize=13,
                        path_effects=[
                            patheffects.withStroke(linewidth=2, foreground='black')
                        ]
                    )
    
    def create_parameter_visualization(self, image_data: ImageData, 
                                     param_info: Dict) -> plt.Figure:
        """Create visualization showing parameter values on the data image."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # Plot data image
        self._plot_data_image(ax, image_data, "Parameter Visualization")
        
        # Get model center relative to cutout
        center_x = image_data.size  # Center of cutout
        center_y = image_data.size  # Center of cutout
        
        # Plot center cross
        ax.plot(center_x, center_y, "+", color="red", markersize=15, markeredgewidth=2)
        ax.text(center_x + 2, center_y + 2, "Center", color="red", 
               fontweight="bold", fontsize=8)
        
        # Plot r0 circles for beta components
        colors = ["yellow", "orange", "purple", "brown", "pink"]
        linestyles = ["--", ":", "-.", "-", "--"]
        
        # Get pixel scale for arcsec conversion
        pixel_scale = image_data.get_pixel_scale_arcsec() if hasattr(image_data, 'get_pixel_scale_arcsec') else 0.492
        
        for i, (comp_name, color, linestyle) in enumerate(zip(['b1', 'b2', 'b3', 'b4', 'b5'], 
                                                             colors, linestyles)):
            if f'{comp_name}.r0' in param_info:
                r0_val = param_info[f'{comp_name}.r0']['val']  # in pixels
                r0_arcsec = r0_val * pixel_scale  # convert to arcseconds
                r0_circle = Circle((center_x, center_y), r0_val, 
                                 fill=False, color=color, linewidth=2, linestyle=linestyle)
                ax.add_patch(r0_circle)
                ax.text(center_x + r0_val + 2, center_y - i*10, 
                       f'r0_{i+1}={r0_arcsec:.1f}"', 
                       color=color, fontweight="bold", fontsize=8)
        
        ax.set_title("Parameter Visualization")
        return fig
    
    def create_corner_plot(self, df, galaxy_name: str, config: FittingConfig, fit_statistics: Optional[np.ndarray] = None, burn_in_length: int = 0, show_plot: bool = True) -> plt.Figure:
        """Create corner plot with medians, quantiles, and improved visualization."""
        param_names = df.columns
        n_params = len(param_names)
        
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        
        param_stats = {}
        param_limits = {}
        for param in param_names:
            q16, q50, q84 = np.percentile(df[param], [16, 50, 84])
            param_stats[param] = {
                'median': q50,
                'q16': q16,
                'q84': q84,
                'minus_err': q50 - q16,
                'plus_err': q84 - q50
            }
            # Calculate consistent limits for both histograms and scatter plots
            data_min, data_max = df[param].min(), df[param].max()
            data_range = data_max - data_min
            param_limits[param] = {
                'min': data_min - 0.1 * data_range,
                'max': data_max + 0.1 * data_range
            }

        if n_params == 1:
            axes = np.array([[axes]])
        elif n_params == 2 and axes.ndim == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j] if n_params > 1 else axes[0, 0]

                if i < j:
                    ax.set_visible(False)
                    continue

                if i == j:
                    param = param_names[i]
                    data = df[param]
                    n_bins = min(50, int(np.sqrt(len(data))))
                    ax.hist(data, bins=n_bins, alpha=0.7, density=True, color='C0', edgecolor='black', linewidth=0.5)
                    
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, 'r-', linewidth=1.5, alpha=0.8)
                    
                    median = param_stats[param]['median']
                    q16 = param_stats[param]['q16']
                    q84 = param_stats[param]['q84']
                    ax.axvline(median, color='black', linestyle='-', linewidth=1.5)
                    ax.axvline(q16, color='orange', linestyle='--', linewidth=2, alpha=0.8)
                    ax.axvline(q84, color='orange', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Set x-limits to match the pairplot limits
                    ax.set_xlim(param_limits[param]['min'], param_limits[param]['max'])
                    
                    minus_err = param_stats[param]['minus_err']
                    plus_err = param_stats[param]['plus_err']
                    formatted_uncertainty = self._format_value_with_uncertainty(median, plus_err, minus_err)
                    param_text = f"$\mathbf{{{param}}} = {formatted_uncertainty}$"
                    ax.set_title(param_text, fontsize=9, pad=5)
                    ax.set_yticks([])
                    ax.set_ylabel('')

                else:
                    param_x = param_names[j]
                    param_y = param_names[i]
                    x_data = df[param_x]
                    y_data = df[param_y]
                    ax.scatter(x_data, y_data, alpha=0.3, s=4.5, color='C0')

                    try:
                        xy = np.vstack([x_data, y_data])
                        kde = gaussian_kde(xy)
                        # Use consistent limits from param_limits
                        x_min = param_limits[param_x]['min']
                        x_max = param_limits[param_x]['max']
                        y_min = param_limits[param_y]['min']
                        y_max = param_limits[param_y]['max']
                        x_grid = np.linspace(x_min, x_max, 50)
                        y_grid = np.linspace(y_min, y_max, 50)
                        X, Y = np.meshgrid(x_grid, y_grid)
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        Z = kde(positions).reshape(X.shape)
                        
                        levels = [np.percentile(Z, 68), np.percentile(Z, 90), np.percentile(Z, 95), Z.max()]
                        viridis_colors = [plt.cm.viridis(0.3), plt.cm.viridis(0.6), plt.cm.viridis(0.9)]
                        ax.contourf(X, Y, Z, levels=levels, colors=viridis_colors, alpha=0.5)
                    except Exception as e:
                        print(f"Could not create contours for {param_x} vs {param_y}: {e}")

                    median_x = param_stats[param_x]['median']
                    median_y = param_stats[param_y]['median']
                    ax.axvline(median_x, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
                    ax.axhline(median_y, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
                    
                    # Set consistent limits for scatter plots
                    ax.set_xlim(param_limits[param_x]['min'], param_limits[param_x]['max'])
                    ax.set_ylim(param_limits[param_y]['min'], param_limits[param_y]['max'])

                if i < n_params - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(param_names[j], fontsize=12)

                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(param_names[i], fontsize=12)

                ax.grid(True, alpha=0.3)

        # Add fit statistics subplot if provided
        if fit_statistics is not None:
            # Add a small subplot in the upper right corner for fit statistics
            # Final tiny left adjustment for perfect alignment
            stats_ax = fig.add_axes([0.57, 0.79, 0.33, 0.14])  # [left, bottom, width, height] - adjusted position

            # Initialize default values
            ylabel = 'Statistic'
            iterations = np.array([])
            
            # Check if fit_statistics is a tuple containing both stats and accept arrays
            if isinstance(fit_statistics, tuple) and len(fit_statistics) == 2:
                # New format: (full_stats, accept_vector)
                full_stats, accept_vector = fit_statistics

                # Calculate reduced statistics
                try:
                    from sherpa.astro.ui import get_fit_results
                    fit_results = get_fit_results()
                    dof = fit_results.dof if hasattr(fit_results, 'dof') and fit_results.dof > 0 else 1

                    reduced_stats = full_stats / dof
                    ylabel = 'Reduced Statistic'
                except:
                    reduced_stats = full_stats
                    ylabel = 'Fit Statistic'

                # Plot all samples, accepted and rejected
                iterations = np.arange(len(reduced_stats))
                stats_ax.plot(iterations, reduced_stats, color='C0', linewidth=1.5, alpha=1, label='All samples')
                stats_ax.plot(iterations[:burn_in_length], reduced_stats[:burn_in_length], color='C1', linewidth=1.5, alpha=1, label='Burn-in')

                # Plot only accepted samples for highlight
                accepted_iterations = iterations[accept_vector.astype(bool)]
                stats_ax.plot(accepted_iterations, reduced_stats[accept_vector.astype(bool)], 'go', markersize=2, alpha=1, label='Accepted')

                # Add burn-in vertical line (optional, if needed)
                stats_ax.axvline(burn_in_length, color='C1', linestyle='--', linewidth=1.5, alpha=0.7)
                
                # Set x-axis to show all sample indices
                if len(iterations) > 0:
                    stats_ax.set_xlim(0, len(iterations) - 1)
            else:
                # Handle legacy or improper format for fit_statistics
                print("Warning: fit_statistics did not match expected format, skipping trace plot.")

            stats_ax.set_xlabel('Iteration Index', fontsize=11)
            stats_ax.set_ylabel(ylabel, fontsize=11)
            stats_ax.set_title('MCMC Trace', fontsize=11, pad=3)
            stats_ax.grid(True, alpha=0.3)
            stats_ax.tick_params(labelsize=10)

            # Set axis limits with some padding based on the data
            # Get all y-values for proper limits
            all_y_values = []
            for line in stats_ax.get_lines():
                all_y_values.extend(line.get_ydata())

            # if len(all_y_values) > 0:
            #     y_min, y_max = min(all_y_values), max(all_y_values)
            #     y_range = y_max - y_min
            #     # Use 5% padding or a minimum padding to prevent too tight limits
            #     y_padding = max(y_range * 0.05, abs(y_min) * 0.01) if y_range > 0 else abs(y_min) * 0.1 + 1
            #     stats_ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # plt.tight_layout()  # Commented out - not compatible with add_axes
        plt.subplots_adjust(top=0.93, hspace=0.05, wspace=0.05)
        if show_plot:
            plt.show(block=False)
        return fig

    def _format_value_with_uncertainty(self, value, plus_err, minus_err):
        max_err = max(plus_err, minus_err)
        if max_err == 0:
            return f"{value:.4f}"
        err_magnitude = math.floor(math.log10(max_err))
        precision = max(0, 1 - err_magnitude)
        plus_rounded = round(plus_err, precision)
        minus_rounded = round(minus_err, precision)
        value_rounded = round(value, precision)
        if precision == 0:
            if abs(plus_rounded - minus_rounded) / max(plus_rounded, minus_rounded) < 0.1:
                avg_err = (plus_rounded + minus_rounded) / 2
                return f"{value_rounded:.0f} \u00b1 {avg_err:.0f}"
            else:
                return f"{value_rounded:.0f}^{{+{plus_rounded:.0f}}}_{{-{minus_rounded:.0f}}}"
        else:
            if abs(plus_rounded - minus_rounded) / max(plus_rounded, minus_rounded) < 0.1:
                avg_err = (plus_rounded + minus_rounded) / 2
                return f"{value_rounded:.{precision}f} \u00b1 {avg_err:.{precision}f}"
            else:
                return f"{value_rounded:.{precision}f}^{{+{plus_rounded:.{precision}f}}}_{{-{minus_rounded:.{precision}f}}}"
