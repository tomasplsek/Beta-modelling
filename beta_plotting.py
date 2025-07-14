#!/usr/bin/env python3
"""
Beta Profile Plotting Module
============================

This module contains purely matplotlib-based plotting functions that consume
data structures from the beta_modeling module. It has no dependencies on Sherpa.

Author: Extracted from beta_fitting_improved.py
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Ellipse, Circle
from matplotlib import gridspec
from matplotlib import patheffects

# Disable astropy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
warnings.filterwarnings('ignore', module='astropy')
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)
from astropy.convolution import Gaussian2DKernel, convolve

# Import data classes from modeling module
from beta_modeling import RadialProfileData, ImageData, FittingConfig


class PlottingManager:
    """Handle all plotting operations using only matplotlib."""
    
    def __init__(self, config: FittingConfig):
        self.config = config
    
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
        
        # Plot data image if available
        if image_data is not None:
            self._plot_data_image(ax_data, image_data, galaxy_name, beta_params, distance_scale)
            ax_data.set_title("Data", fontsize=10)
            
            # Plot model and residual images
            self._plot_model_image(ax_model, image_data, beta_params, distance_scale)
            ax_model.set_title("Model", fontsize=10)
            
            self._plot_residual_image(ax_residual_img, image_data, beta_params, distance_scale)
            ax_residual_img.set_title("Residual", fontsize=10)
        else:
            # No image data available
            for ax, title in zip([ax_data, ax_model, ax_residual_img], 
                               ["Data", "Model", "Residual"]):
                ax.text(0.5, 0.5, 'No image\ndata available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(title, fontsize=10)
        
        # Plot radial profile if available
        if profile_data is not None:
            self._plot_radial_profile_with_residual(ax_radial, ax_residual_plot, 
                                                   profile_data, fitted)
        else:
            # No profile data
            ax_radial.text(0.5, 0.5, 'No radial\nprofile available', 
                          ha='center', va='center', transform=ax_radial.transAxes, fontsize=10)
            ax_residual_plot.text(0.5, 0.5, 'No data', 
                                 ha='center', va='center', transform=ax_residual_plot.transAxes, fontsize=10)
        
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
                                         fitted: bool = False):
        """Plot radial profile with residual subplot below, including individual components."""
        # Conversion factor
        c = profile_data.pixel_scale ** 2
        
        # Plot data and total fit in main subplot (using pixels for x-axis)
        good_data = profile_data.data_error / profile_data.data < 2
        ax_main.errorbar(profile_data.radius_pixels[good_data], 
                        profile_data.data[good_data] / c, 
                        yerr=profile_data.data_error[good_data] / c, 
                        fmt='o', markersize=2, capsize=1, label='Data', alpha=0.7)
        ax_main.plot(profile_data.radius_pixels, profile_data.model / c, 'r-', 
                    linewidth=2, label='Total Model')
        
        # Plot individual components
        for comp_name, comp_data in profile_data.components.items():
            if comp_name == 'b1':
                ax_main.plot(profile_data.radius_pixels, comp_data / c, '--', 
                           color='blue', linewidth=1.5, 
                           label=f'Beta 1', alpha=0.8)
            elif comp_name == 'b2':
                ax_main.plot(profile_data.radius_pixels, comp_data / c, ':', 
                           color='green', linewidth=1.5, 
                           label=f'Beta 2', alpha=0.8)
            elif comp_name == 'b3':
                ax_main.plot(profile_data.radius_pixels, comp_data / c, '-.', 
                           color='purple', linewidth=1.5, 
                           label=f'Beta 3', alpha=0.8)
            elif comp_name == 'b4':
                ax_main.plot(profile_data.radius_pixels, comp_data / c, '-', 
                           color='brown', linewidth=1.5, 
                           label=f'Beta 4', alpha=0.8)
            elif comp_name == 'b5':
                ax_main.plot(profile_data.radius_pixels, comp_data / c, '--', 
                           color='pink', linewidth=1.5, 
                           label=f'Beta 5', alpha=0.8)
            elif comp_name == 'bkg':
                ax_main.axhline(comp_data[0] / c, color='gray', linestyle='-', 
                              linewidth=1, alpha=0.7, 
                              label=f'Background ({comp_data[0]:.2e})')
        
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_ylabel("Counts/pixel")
        ax_main.legend(fontsize=7, loc='upper right')
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelsize=8)
        ax_main.set_xlabel("Radius (pixels)", fontsize=8)
        
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
        ax_top.set_xlabel("Radius (arcsec)", fontsize=8)
        ax_top.tick_params(labelsize=8)

        # Plot residuals in bottom subplot (using pixels)
        residuals = profile_data.get_residuals()
        ax_res.errorbar(profile_data.radius_pixels[good_data], residuals[good_data], 
                       yerr=1, fmt='o', markersize=2, capsize=1, alpha=0.7)
        ax_res.axhline(0, ls='--', color='black', alpha=0.5)
        ax_res.set_ylabel("Residual (σ)", fontsize=8)
        
        ax_res.set_xscale("log")
        ax_res.set_xlabel("Radius (pixels)", fontsize=8)
        ax_res.tick_params(labelsize=8)
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
        
        # Plot with log scale
        ax.imshow(np.log10(data_cutout + 1), origin="lower")
        
        # Add basic annotations
        self._add_image_annotations_data_only(ax, image_data.scale)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale)
        
        # Add angular scale bar
        self._add_angular_scale_bar(ax, image_data, distance_scale=distance_scale)
    
    def _plot_model_image(self, ax, image_data: ImageData, 
                         beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                         distance_scale: Optional[float] = None):
        """Plot the model image."""
        model_cutout = image_data.model_cutout
        
        cmap = plt.cm.viridis
        ax.imshow(model_cutout + 1, origin="lower", norm=LogNorm(), cmap=cmap)
        self._add_image_annotations(ax, image_data.scale)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale)
        
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
        
        ax.imshow(residual_cutout, origin="lower", cmap="coolwarm")
        self._add_image_annotations(ax, image_data.scale)
        
        # Add beta model overlays if available
        self._add_beta_model_overlays(ax, image_data, beta_params, distance_scale)
        
        # Add angular scale bar
        self._add_angular_scale_bar(ax, image_data, distance_scale=distance_scale)
    
    def _add_image_annotations(self, ax, scale: float):
        """Add common annotations to image plots."""
        ticks = np.linspace(0, 2*scale // 64 * 64, 5)
        ax.set_xticks(ticks + np.array([0, 0, 0, 0, -1]))
        ax.set_yticks(ticks)
    
    def _add_image_annotations_data_only(self, ax, scale: float):
        """Add annotations to data-only plots (no model ellipses)."""
        ticks = np.linspace(0, 2*scale // 64 * 64, 5)
        ax.set_xticks(ticks + np.array([0, 0, 0, 0, -1]))
        ax.set_yticks(ticks)
    
    def _add_angular_scale_bar(self, ax, image_data: ImageData, scale_arcsec: float = 20.0, distance_scale: Optional[float] = None):
        """Add angular scale bar to the bottom left of the image."""
        from matplotlib import patheffects
        
        # Get pixel scale
        pixel_scale = image_data.get_pixel_scale_arcsec() if hasattr(image_data, 'get_pixel_scale_arcsec') else 0.492
        
        # Convert scale from arcseconds to pixels
        scale_pixels = scale_arcsec / pixel_scale
        
        # Get image dimensions (cutout)
        cutout_size = 2 * image_data.scale
        
        # Position for scale bar (bottom left, with more margin to accommodate kpc text)
        margin_x = cutout_size * 0.12  # Increased margin from left (moved right)
        # Always move up to make room for kpc text (even if not displayed)
        margin_y = cutout_size * 0.12  # Increased margin from bottom
        
        start_x = margin_x
        start_y = margin_y
        end_x = start_x + scale_pixels
        end_y = start_y
        
        # Draw black outline line (slightly thicker and longer for contrast)
        ax.plot([start_x-2, end_x+2], [start_y, end_y], 'k-', linewidth=5, solid_capstyle='butt')
        # Draw white scale bar line on top
        ax.plot([start_x, end_x], [start_y, end_y], 'w-', linewidth=3, solid_capstyle='butt')
        
        # Add text label for arcseconds
        text_x = start_x + scale_pixels / 2  # Center the text over the line
        text_y = start_y + cutout_size * 0.03  # Slightly above the line
        
        # Add arcsec text with black outline
        ax.text(text_x, text_y, f'{scale_arcsec:.0f} arcsec', 
               color='white', fontweight='bold', fontsize=10, 
               ha='center', va='bottom',
               path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
        
        # Add kpc scale if distance scale is provided
        if distance_scale is not None and distance_scale > 0:
            # Convert arcseconds to kpc: arcsec * pc/arcsec / 1000
            scale_kpc = scale_arcsec * distance_scale / 1000.0
            kpc_text_y = start_y - cutout_size * 0.03  # Below the line
            
            # Add kpc text with black outline
            ax.text(text_x, kpc_text_y, f'{scale_kpc:.1f} kpc', 
                   color='white', fontweight='bold', fontsize=10, 
                   ha='center', va='top',
                   path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    def _add_beta_model_overlays(self, ax, image_data: ImageData, 
                                beta_params: Optional[Dict[str, Dict[str, float]]] = None,
                                distance_scale: Optional[float] = None):
        """Add beta model center markers and core radius circles to the image."""
        if not beta_params:
            return
        
        # Get cutout center coordinates
        cutout_center_x = image_data.scale
        cutout_center_y = image_data.scale
        
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
                cutout_x = model_x - (image_center_x - image_data.scale)
                cutout_y = model_y - (image_center_y - image_data.scale)
                
                # Add center marker (smaller)
                ax.plot(cutout_x, cutout_y, '+', color=color, markersize=6, 
                       markeredgewidth=1.5, label=f'{comp_name.upper()} center')
                
                # Add core radius circle
                if core_radius > 0:
                    circle = Circle((cutout_x, cutout_y), core_radius, 
                                  fill=False, color=color, linewidth=2, 
                                  linestyle=linestyle, alpha=0.8)
                    ax.add_patch(circle)
                    
                    # Add text label for core radius
                    pixel_scale = image_data.get_pixel_scale_arcsec() if hasattr(image_data, 'get_pixel_scale_arcsec') else 0.492
                    text_x = cutout_x + core_radius * 1.0
                    text_y = cutout_y + core_radius * 1.0
                    
                    if distance_scale is not None and distance_scale > 0:
                        # Convert to kpc: pixels -> arcsec -> parsec -> kpc
                        # distance_scale is now pc/arcsec, so multiply instead of divide
                        core_radius_arcsec = core_radius * pixel_scale
                        core_radius_parsec = core_radius_arcsec * distance_scale
                        core_radius_kpc = core_radius_parsec / 1000.0
                        label_text = f'$r_0={core_radius_kpc:.2f}$ kpc'
                    else:
                        # Use arcseconds as before
                        core_radius_arcsec = core_radius * pixel_scale
                        label_text = f'$r_0={core_radius_arcsec:.1f}$"'
                    
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
        center_x = image_data.scale  # Center of cutout
        center_y = image_data.scale  # Center of cutout
        
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
    
    def create_corner_plot(self, df, galaxy_name: str, config: FittingConfig) -> plt.Figure:
        """Create corner plot for MCMC results."""
        try:
            from corner import corner
        except ImportError:
            raise ImportError("Corner package required for MCMC plotting")
        
        fig = plt.figure(figsize=(12, 12))
        
        # Calculate statistics
        q1, q3 = config.quantiles
        table = df.describe(percentiles=[q1, 0.5, q3]).T[["15.9%", "50%", "84.1%"]]
        table["Minus"] = table["50%"] - table["15.9%"]
        table["Plus"] = table["84.1%"] - table["50%"]
        table["Median"] = table["50%"]
        table = table.round(3)
        
        # Create corner plot
        fig = corner(df, color="C0", fig=fig, labels=df.columns,
                    labelpad=0.09, show_titles=True)
        
        # Add quantile lines
        axes = np.array(fig.axes).reshape((len(df.columns), len(df.columns)))
        for i, col in enumerate(df.columns):
            ax = axes[i, i]
            ax.axvline(np.quantile(df[col], q1), color="C2", ls="--", lw=2.5)
            ax.axvline(np.quantile(df[col], 0.5), color="C1", ls="--", lw=2.5)
            ax.axvline(np.quantile(df[col], q3), color="C2", ls="--", lw=2.5)
        
        plt.savefig(f"{galaxy_name}_chain.pdf")
        return fig
