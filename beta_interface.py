#!/usr/bin/env python3
"""
Beta Profile Interface Module
=============================

This module provides the interactive ipywidgets interface that coordinates
functionality between the modeling and plotting modules.

Author: Extracted from beta_fitting_improved.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import utility functions
from utils import suppress_astropy_warnings
suppress_astropy_warnings()

# Jupyter/IPython widgets
import ipywidgets as widgets
from IPython.display import clear_output, display

# Import from our modules
from beta_modeling import BetaFittingTool
from beta_plotting import PlottingManager




def create_beta_fitting_interface():
    """Create an interactive widget interface for beta fitting with main menu layout."""
    tool = BetaFittingTool()
    plotting_manager = PlottingManager(tool.config)
    
    # Cache for managing expensive fetch operations
    class PlotCache:
        def __init__(self):
            self._profile_data = None
            self._image_data = None
            self._beta_params = None
            self._model_hash = None
        
        def get_cached_or_fetch(self, tool):
            current_hash = hash((tool.current_galaxy.name, tool.current_model.model_type))
            if self._model_hash != current_hash:
                self._refresh_cache(tool)
                self._model_hash = current_hash
            return self._profile_data, self._image_data, self._beta_params
        
        def _refresh_cache(self, tool):
            self._profile_data = tool.get_radial_profile_data()
            self._image_data = tool.get_image_data()
            self._beta_params = tool.get_beta_model_params()
        
        def invalidate_cache(self):
            """Force cache refresh on next access."""
            self._model_hash = None
    
    cache = PlotCache()
    
    # Global variables to store parameter sliders and MCMC results
    param_sliders = {}
    current_mcmc_results = None
    current_fit_stats = None
    
    # Output areas
    output_area = widgets.Output()
    plot_area = widgets.Output()
    sliders_area = widgets.Output()
    
    # Main menu widgets
    # Set NGC4649.fits as default, fallback to first file if not found
    default_galaxy = next((f.name for f in tool.galaxy_files if f.name == 'NGC4649.fits'), tool.galaxy_files[0].name if tool.galaxy_files else '')
    
    galaxy_dropdown = widgets.Dropdown(
        options=[f.name for f in tool.galaxy_files],
        value=default_galaxy,
        description="Galaxy: ",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    model_dropdown = widgets.Dropdown(
        options=tool.config.available_models,
        value=tool.config.available_models[0],
        description="Model:   ",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    size_dropdown = widgets.Dropdown(
        options=tool.config.available_scales,
        value="original",
        description="Size:    ",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    method_dropdown = widgets.Dropdown(
        options=tool.config.available_methods,
        value=tool.config.default_method,
        description="Method: ",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    statistic_dropdown = widgets.Dropdown(
        options=tool.config.available_statistics,
        value=tool.config.default_statistic,
        description="Statistic:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    # Scale input (pc/arcsec conversion)
    scale_input = widgets.FloatText(
        value=0.0,
        placeholder="Enter pc/arcsec",
        description="Scale (pc/arcsec):",
        tooltip="Scale in pc/arcsec (0 = disabled)",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    # MCMC controls
    mcmc_length_input = widgets.IntText(
        value=tool.config.default_mcmc_length,
        step=100,
        description="MCMC Length:",
        tooltip="Number of MCMC iterations",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    mcmc_burnin_input = widgets.IntText(
        value=tool.config.default_burn_in,
        step=100,
        description="Burn-in:",
        tooltip="Number of burn-in iterations to discard",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='220px')
    )
    
    # Buttons
    fit_button = widgets.Button(description="Fit Model", button_style="success", layout=widgets.Layout(width='100px'))
    mcmc_button = widgets.Button(description="Run MCMC", button_style="danger", layout=widgets.Layout(width='100px'))
    save_corner_button = widgets.Button(description="Save Corner", button_style="info", layout=widgets.Layout(width='100px'))
    save_params_button = widgets.Button(description="Save Params", button_style="info", layout=widgets.Layout(width='100px'))
    save_residual_button = widgets.Button(description="Save Resid", button_style="info", layout=widgets.Layout(width='100px'))
    save_model_button = widgets.Button(description="Save Model", button_style="info", layout=widgets.Layout(width='100px'))
    save_combined_button = widgets.Button(description="Save Plot", button_style="info", layout=widgets.Layout(width='100px'))
    
    # Status display
    status = widgets.HTML(value="<b>Status:</b> Ready")
    
    def update_size_dropdown_options():
        """Update size dropdown options based on current galaxy image size."""
        if tool.current_galaxy is None:
            # No galaxy loaded, use all available scales
            size_dropdown.options = tool.config.available_scales
            return
        
        # Get current galaxy dimensions
        img_height, img_width = tool.current_galaxy.shape
        min_dimension = min(img_height, img_width)

        # Generate multiples of 128 up to min_dimension
        valid_scales = ["original"]  # Always include original
        
        for scale_val in range(128, min_dimension + 1, 128):
            valid_scales.append(f"{scale_val} x {scale_val} pixels")
        
        # Update dropdown options
        current_value = size_dropdown.value
        size_dropdown.options = valid_scales
        
        # Reset to "original" if current selection is no longer valid
        if current_value not in valid_scales:
            size_dropdown.value = "original"
            
    def create_parameter_sliders():
        """Create sliders for model parameters with log scale for specific parameters, freeze checkboxes, and linking checkboxes."""
        nonlocal param_sliders
        
        with sliders_area:
            clear_output()
            
            if tool.current_model is None:
                print("Load a galaxy first to see parameters")
                return
            
            param_info = tool.get_parameter_info()
            param_sliders = {}
            
            # Parameters that should use log scale
            log_scale_params = ['b1.r0', 'b1.ampl', 'b1.alpha', 'b2.r0', 'b2.ampl', 'b2.alpha', 
                               'b3.r0', 'b3.ampl', 'b3.alpha', 'b4.r0', 'b4.ampl', 'b4.alpha',
                               'b5.r0', 'b5.ampl', 'b5.alpha', 'g1.ampl', 'bkg.c0']
            
            
            def clean_parameter_name(param_name):
                """Remove beta2d. and const2d. prefixes from parameter names."""
                clean_name = param_name
                if 'beta2d.' in clean_name:
                    clean_name = clean_name.replace('beta2d.', '')
                if 'const2d.' in clean_name:
                    clean_name = clean_name.replace('const2d.', '')
                return clean_name
            
            def create_parameter_widgets(param_name, info):
                """Create widgets for a single parameter to ensure independence."""
                # Clean parameter name for display
                display_name = clean_parameter_name(info['name'])
                
                # Determine if this parameter should use log scale
                use_log_scale = param_name in log_scale_params
                
                if use_log_scale:
                    # Use log slider for specified parameters
                    min_val = max(info['min'], 1e-6)  # Avoid log(0)
                    max_val = info['max']
                    current_val = max(info['val'], min_val)
                    
                    slider = widgets.FloatLogSlider(
                        value=current_val,
                        base=10,
                        min=np.log10(min_val),
                        max=np.log10(max_val),
                        step=0.01,
                        description=display_name,
                        layout=widgets.Layout(width='87%')
                    )
                else:
                    # Use regular slider for other parameters
                    slider = widgets.FloatSlider(
                        value=info['val'],
                        min=info['min'],
                        max=info['max'],
                        step=(info['max'] - info['min']) / 100,
                        description=display_name,
                        layout=widgets.Layout(width='87%')
                    )
                
                # Create completely independent freeze checkbox for this specific parameter
                freeze_checkbox = widgets.Checkbox(
                    value=info['frozen'],
                    description="", #Freeze",
                    layout=widgets.Layout(width='8%'),
                    tooltip=f'Freeze/unfreeze {param_name}',
                    indent=False
                )
                
                # Connect slider to update function with proper closure capture
                def make_slider_handler(pname):
                    def slider_changed(change):
                        if tool.current_model is not None and not _updating_from_backend:
                            tool.update_parameter(pname, change['new'])
                            cache.invalidate_cache()
                            update_plot_with_current_model()
                    return slider_changed
                
                slider.observe(make_slider_handler(param_name), names='value')
                
                # Connect checkbox to freeze function with proper closure capture
                def make_freeze_handler(pname):
                    def freeze_param(change):
                        if tool.current_model is not None:
                            try:
                                tool.freeze_parameter(pname, freeze_param=change['new'])
                                with output_area:
                                    print(f"✓ Parameter {pname} {'frozen' if change['new'] else 'unfrozen'}")
                            except Exception as e:
                                with output_area:
                                    print(f"✗ Error freezing/unfreezing {pname}: {e}")
                    return freeze_param
                
                freeze_checkbox.observe(make_freeze_handler(param_name), names='value')
                
                return slider, freeze_checkbox
            
            if param_info:
                # print("Model Parameters:")
                
                # Add "Freeze" label above the first parameter
                # Create a header row with "Freeze" aligned to the right
                freeze_header = widgets.HTML(
                    value='<div style="display: flex; justify-content: space-between; width: 100%; margin-bottom: 5px;"><span><b>Model parameters</b></span><span style="font-weight: bold;">Freeze</span></div>',
                    layout=widgets.Layout(width='98%')
                )
                display(freeze_header)
                
                for param_name, info in param_info.items():
                    # Create widgets for this parameter
                    slider, freeze_checkbox = create_parameter_widgets(param_name, info)
                    
                    # Remove "Freeze" description from checkboxes since we have a header now
                    freeze_checkbox.description = ""
                    
                    # Store widgets
                    param_sliders[param_name] = {
                        'slider': slider, 
                        'checkbox': freeze_checkbox
                    }
                    
                    # Create horizontal layout with slider and checkbox
                    param_row = widgets.HBox([slider, freeze_checkbox], 
                                           layout=widgets.Layout(width='100%', justify_content='space-between'))
                    display(param_row)

    # Flag to prevent slider callbacks from triggering plot updates during backend sync
    _updating_from_backend = False
    
    def update_sliders_from_backend():
        """Update slider values and checkbox states from the backend parameter values."""
        nonlocal _updating_from_backend
        if tool.current_model is None:
            return
        
        try:
            # Set flag to prevent callbacks from triggering plot updates
            _updating_from_backend = True
            
            param_info = tool.get_parameter_info()
            for param_name, info in param_info.items():
                if param_name in param_sliders:
                    # Update slider value
                    param_sliders[param_name]['slider'].value = info['val']
                    # Update freeze checkbox
                    param_sliders[param_name]['checkbox'].value = info['frozen']
        except Exception as e:
            print(f"Error updating sliders from backend: {e}")
        finally:
            # Always reset the flag
            _updating_from_backend = False
    
    # Debouncing mechanism for plot updates
    import threading
    _plot_update_timer = None
    
    def debounced_update_plot_with_current_model():
        """Debounced version of plot update - delays plot updates to avoid excessive redraws."""
        nonlocal _plot_update_timer
        
        # Cancel existing timer if one is running
        if _plot_update_timer is not None:
            _plot_update_timer.cancel()
        
        # Start new timer - plot will update after 300ms of no slider activity
        _plot_update_timer = threading.Timer(0.3, update_plot_with_current_model)
        _plot_update_timer.start()
    
    def update_plot_with_current_model():
        """Update plot with current model parameters."""
        try:
            with plot_area:
                clear_output(wait=True)
                
                # Get data from modeling module
                profile_data, image_data, beta_params = cache.get_cached_or_fetch(tool)
                
                # Check if we have fit results
                try:
                    from sherpa.astro.ui import get_fit_results
                    get_fit_results()
                    # If we get here, model has been fitted
                    fitted = True
                except:
                    # Model not fitted yet, but show current model
                    fitted = False
                
                # Get scale for kpc conversion (pc/arcsec)
                pc_per_arcsec_scale = scale_input.value if scale_input.value > 0 else None
                
                # Create plot using plotting manager
                fig = plotting_manager.create_comprehensive_plot(
                    profile_data, image_data, tool.current_galaxy.name,
                    tool.current_model.model_type if tool.current_model else "",
                    fitted, beta_params, distance_scale=pc_per_arcsec_scale
                )
                plt.show()
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    # Event handlers
    def on_fit_model(b):
        with output_area:
            clear_output()
            if tool.current_model is None:
                print("Please load a galaxy first!")
                return
            
            try:
                print(f"Fitting {tool.current_model.model_type} model...")
                tool.fit_model(method=method_dropdown.value, statistic=statistic_dropdown.value)
                status.value = "<b>Status:</b> ✅ Fit completed"
                print("\nFitting completed!")
                
                # Update sliders with fitted values
                update_sliders_from_backend()
                
                # Show frozen status
                param_info = tool.get_parameter_info()
                frozen_params = [p for p, info in param_info.items() if info['frozen']]
                if frozen_params:
                    print(f"\nFrozen parameters during fit: {', '.join(frozen_params)}")
                else:
                    print("\nNo parameters were frozen during the fit")
                
                # Update plot with fit results
                update_plot_with_current_model()
                    
            except Exception as e:
                status.value = f"<b>Status:</b> ❌ Error: {str(e)[:50]}..."
                print(f"Error: {e}")
    
    def on_model_change(change):
        """Handle model dropdown changes"""
        nonlocal current_mcmc_results, current_fit_stats
        if tool.current_galaxy is not None:
            with output_area:
                clear_output()
                try:
                    print(f"Changing to {change['new']} model...")
                    tool.setup_model(change['new'])
                    cache.invalidate_cache()
                    
                    # Clear MCMC results when model changes
                    current_mcmc_results = None
                    current_fit_stats = None
                    print("MCMC results cleared due to model change.")
                    
                    status.value = f"<b>Status:</b> ✅ Model: {tool.current_model.model_type}"
                    print("Model changed!")
                    
                    # Recreate parameter sliders
                    create_parameter_sliders()
                    
                    # Sync sliders with backend state
                    update_sliders_from_backend()
                    
                    # Update plot with new unfitted model
                    update_plot_with_current_model()
                        
                except Exception as e:
                    status.value = f"<b>Status:</b> ❌ Error: {str(e)[:50]}..."
                    print(f"Error: {e}")
    
    def on_galaxy_change(change):
        """Handle galaxy dropdown changes"""
        with output_area:
            clear_output()
            try:
                galaxy_file = next(f for f in tool.galaxy_files if f.name == change['new'])
                print(f"Loading: {galaxy_file.name}")
                tool.load_galaxy(galaxy_file, scale=size_dropdown.value)
                cache.invalidate_cache()
                status.value = f"<b>Status:</b> ✅ Loaded: {tool.current_galaxy.name}"
                
                # Update size dropdown options based on new galaxy size
                update_size_dropdown_options()
                
                # Auto-setup current model type
                current_model_type = model_dropdown.value
                print(f"Setting up {current_model_type} model...")
                tool.setup_model(current_model_type)
                print("Model setup completed!")
                status.value = f"<b>Status:</b> ✅ Model: {current_model_type} ready"
                
                # Create parameter sliders
                create_parameter_sliders()
                
                # Sync sliders with backend state
                update_sliders_from_backend()
                
                # Show initial frozen status
                param_info = tool.get_parameter_info()
                frozen_params = [p for p, info in param_info.items() if info['frozen']]
                if frozen_params:
                    print(f"\nInitially frozen parameters: {', '.join(frozen_params)}")
                
                # Create comprehensive plot showing unfitted model
                update_plot_with_current_model()
                    
            except Exception as e:
                status.value = f"<b>Status:</b> ❌ Error: {str(e)[:50]}..."
                print(f"Error: {e}")
    
    def on_save_parameters(b):
        """Save fit parameters to text file"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No model or galaxy loaded!")
                    return
                
                # Get current parameter info
                param_info = tool.get_parameter_info()
                galaxy_name = tool.current_galaxy.name
                model_type = tool.current_model.model_type
                
                # Create output folder based on galaxy name (without .fits)
                folder_name = galaxy_name.replace('.fits', '')
                import os
                os.makedirs(folder_name, exist_ok=True)
                
                # Create output filename in the folder
                filename = os.path.join(folder_name, f"{galaxy_name}_{model_type.replace(' ', '_')}_params.txt")
                
                # Write parameters to file
                with open(filename, 'w') as f:
                    f.write(f"Beta Profile Fit Parameters\n")
                    f.write(f"Galaxy: {galaxy_name}\n")
                    f.write(f"Model: {model_type}\n")
                    f.write(f"Size: {tool.current_scale}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    for param_name, info in param_info.items():
                        status_str = "FROZEN" if info['frozen'] else "FREE"
                        f.write(f"{info['name']}: {info['val']:.6e}\n")
                        f.write(f"  Range: [{info['min']:.3e}, {info['max']:.3e}]\n")
                        f.write(f"  Status: {status_str}\n\n")
                    
                    # Add fit statistics if available
                    try:
                        from sherpa.astro.ui import get_fit_results
                        fit_results = get_fit_results()
                        f.write(f"Fit Statistics:\n")
                        f.write(f"Statistic: {fit_results.statname}\n")
                        f.write(f"Statistic value: {fit_results.statval:.6e}\n")
                        f.write(f"Degrees of freedom: {fit_results.dof}\n")
                        f.write(f"Reduced statistic: {fit_results.rstat:.6e}\n")
                    except:
                        f.write(f"Fit Statistics: Not available (model not fitted)\n")
                
                print(f"Parameters saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving parameters: {e}")
    
    def on_save_residual(b):
        """Save residual image with original header, properly cropped to scale"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No fitted model or galaxy loaded!")
                    return
                
                from sherpa.astro.ui import get_resid_image, get_data
                from astropy.io import fits
                import numpy as np
                import os
                
                # Get residual data
                resid = get_resid_image().y.reshape(get_data().shape)
                
                # Crop to actual scale if not original
                if tool.current_scale != "original":
                    # Extract scale value (e.g., "128 x 128 pixels" -> 128)
                    scale_val = int(tool.current_scale.split()[0].split("x")[0])
                    center = tool.current_galaxy.center
                    
                    # Calculate crop boundaries
                    x_center, y_center = int(center[0]), int(center[1])
                    half_scale = scale_val // 2
                    
                    x_start = max(0, x_center - half_scale)
                    x_end = min(resid.shape[0], x_center + half_scale)
                    y_start = max(0, y_center - half_scale)
                    y_end = min(resid.shape[1], y_center + half_scale)
                    
                    # Crop the residual data
                    resid_cropped = resid[x_start:x_end, y_start:y_end]
                    
                    # Update header with new dimensions
                    original_header = tool.current_galaxy.header.copy()
                    original_header['NAXIS1'] = resid_cropped.shape[1]
                    original_header['NAXIS2'] = resid_cropped.shape[0]
                    
                    # Update reference pixel position if CRPIX keywords exist
                    if 'CRPIX1' in original_header:
                        original_header['CRPIX1'] = original_header['CRPIX1'] - y_start
                    if 'CRPIX2' in original_header:
                        original_header['CRPIX2'] = original_header['CRPIX2'] - x_start
                    
                    print(f"Cropped residual from {resid.shape} to {resid_cropped.shape}")
                    resid = resid_cropped
                else:
                    original_header = tool.current_galaxy.header.copy()
                
                # Create output folder based on galaxy name (without .fits)
                galaxy_name = tool.current_galaxy.name
                folder_name = galaxy_name.replace('.fits', '')
                os.makedirs(folder_name, exist_ok=True)
                
                # Create filename in the folder
                model_type = tool.current_model.model_type
                scale_suffix = f"_{tool.current_scale.replace(' ', '_').replace('x', 'x')}" if tool.current_scale != "original" else ""
                filename = os.path.join(folder_name, f"{galaxy_name}_{model_type.replace(' ', '_')}_residual{scale_suffix}.fits")
                
                # Save FITS file with updated header
                hdu = fits.PrimaryHDU(data=resid, header=original_header)
                hdu.writeto(filename, overwrite=True)
                
                print(f"Residual image saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving residual image: {e}")
    
    def on_save_model(b):
        """Save model image with original header, properly cropped to scale"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No model or galaxy loaded!")
                    return
                
                from sherpa.astro.ui import get_model_image, get_data
                from astropy.io import fits
                import numpy as np
                import os
                
                # Get model data
                model_data = get_model_image().y.reshape(get_data().shape)
                
                # Crop to actual scale if not original
                if tool.current_scale != "original":
                    # Extract scale value (e.g., "128 x 128 pixels" -> 128)
                    scale_val = int(tool.current_scale.split()[0].split("x")[0])
                    center = tool.current_galaxy.center
                    
                    # Calculate crop boundaries
                    x_center, y_center = int(center[0]), int(center[1])
                    half_scale = scale_val // 2
                    
                    x_start = max(0, x_center - half_scale)
                    x_end = min(model_data.shape[0], x_center + half_scale)
                    y_start = max(0, y_center - half_scale)
                    y_end = min(model_data.shape[1], y_center + half_scale)
                    
                    # Crop the model data
                    model_cropped = model_data[x_start:x_end, y_start:y_end]
                    
                    # Update header with new dimensions
                    original_header = tool.current_galaxy.header.copy()
                    original_header['NAXIS1'] = model_cropped.shape[1]
                    original_header['NAXIS2'] = model_cropped.shape[0]
                    
                    # Update reference pixel position if CRPIX keywords exist
                    if 'CRPIX1' in original_header:
                        original_header['CRPIX1'] = original_header['CRPIX1'] - y_start
                    if 'CRPIX2' in original_header:
                        original_header['CRPIX2'] = original_header['CRPIX2'] - x_start
                    
                    print(f"Cropped model from {model_data.shape} to {model_cropped.shape}")
                    model_data = model_cropped
                else:
                    original_header = tool.current_galaxy.header.copy()
                
                # Create output folder based on galaxy name (without .fits)
                galaxy_name = tool.current_galaxy.name
                folder_name = galaxy_name.replace('.fits', '')
                os.makedirs(folder_name, exist_ok=True)
                
                # Create filename in the folder
                model_type = tool.current_model.model_type
                scale_suffix = f"_{tool.current_scale.replace(' ', '_').replace('x', 'x')}" if tool.current_scale != "original" else ""
                filename = os.path.join(folder_name, f"{galaxy_name}_{model_type.replace(' ', '_')}_model{scale_suffix}.fits")
                
                # Save FITS file with updated header
                hdu = fits.PrimaryHDU(data=model_data, header=original_header)
                hdu.writeto(filename, overwrite=True)
                
                print(f"Model image saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving model image: {e}")
    
    def on_scale_change(change):
        """Handle scale input changes (pc/arcsec)"""
        if tool.current_galaxy is not None and tool.current_model is not None:
            with output_area:
                try:
                    scale_value = change['new']
                    if scale_value > 0:
                        print(f"Scale set to {scale_value:.3f} pc/arcsec")
                        print("Core radii will be displayed in kpc")
                    else:
                        print("Scale disabled - core radii in arcsec")
                    
                    # Update plot with new scale
                    update_plot_with_current_model()
                        
                except Exception as e:
                    print(f"Error updating scale: {e}")
    
    def on_scale_dropdown_change(change):
        """Handle scale dropdown changes (image scale)"""
        if tool.current_galaxy is not None:
            with output_area:
                clear_output()
                try:
                    new_scale = change['new']
                    print(f"Changing image scale to: {new_scale}")
                    
                    # Reload galaxy with new scale
                    galaxy_file = tool.current_galaxy.filepath
                    tool.load_galaxy(galaxy_file, scale=new_scale)
                    status.value = f"<b>Status:</b> ✅ Reloaded with scale: {new_scale}"
                    
                    # Re-setup current model type with new scale
                    current_model_type = model_dropdown.value
                    print(f"Re-setting up {current_model_type} model...")
                    tool.setup_model(current_model_type)
                    print("Model setup completed!")
                    status.value = f"<b>Status:</b> ✅ Model: {current_model_type} ready (scale: {new_scale})"
                    
                    # Recreate parameter sliders
                    create_parameter_sliders()
                    
                    # Sync sliders with backend state
                    update_sliders_from_backend()
                    
                    # Show initial frozen status
                    param_info = tool.get_parameter_info()
                    frozen_params = [p for p, info in param_info.items() if info['frozen']]
                    if frozen_params:
                        print(f"\nInitially frozen parameters: {', '.join(frozen_params)}")
                    
                    # Update plot with new scale
                    update_plot_with_current_model()
                        
                except Exception as e:
                    status.value = f"<b>Status:</b> ❌ Error: {str(e)[:50]}..."
                    print(f"Error changing scale: {e}")
    
    def on_run_mcmc(b):
        """Run MCMC analysis on the fitted model"""
        nonlocal current_mcmc_results, current_fit_stats
        with output_area:
            clear_output()
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No model or galaxy loaded!")
                    return
                
                # Check if model has been fitted
                try:
                    from sherpa.astro.ui import get_fit_results
                    fit_results = get_fit_results()
                    print(f"Model fitted with {fit_results.statname} = {fit_results.statval:.3f}")
                except:
                    print("Warning: Model doesn't appear to be fitted. Running MCMC on current parameters.")
                
                # Get MCMC parameters
                mcmc_length = mcmc_length_input.value
                burn_in = mcmc_burnin_input.value
                
                if mcmc_length <= 0 or burn_in < 0:
                    print("Error: MCMC length must be > 0 and burn-in must be >= 0")
                    return
                
                if burn_in >= mcmc_length:
                    print("Error: Burn-in must be less than MCMC length")
                    return
                
                print(f"Starting MCMC analysis...")
                print(f"MCMC Length: {mcmc_length}")
                print(f"Burn-in: {burn_in}")
                print(f"Effective samples: {mcmc_length - burn_in}\n")
                
                status.value = "<b>Status:</b> 🔄 Running MCMC..."
                
                # Run MCMC analysis with fit statistics
                mcmc_results, fit_stats, accept = tool.mcmc_analyzer.run_mcmc(mcmc_length, burn_in, tool.current_galaxy.name, model_type=tool.current_model.model_type, return_stats=True)
                
                # Note: mcmc_results already has burn-in removed by the MCMC analyzer
                # fit_stats contains the full statistics array (including burn-in)
                # accept contains the acceptance array (including burn-in)

                # Store MCMC results globally for saving later
                current_mcmc_results = mcmc_results
                current_fit_stats = (fit_stats, accept)
                
                status.value = "<b>Status:</b> ✅ MCMC completed"
                print(f"\nMCMC analysis completed!")
                print(f"Chain saved as: {tool.current_galaxy.name}_{tool.current_model.model_type.replace(' ', '_')}_chain.csv")
                
                # Display parameter statistics as DataFrame
                import pandas as pd
                
                # Calculate quantiles for each parameter
                param_stats = []
                for col in mcmc_results.columns:
                    q16, q50, q84 = np.percentile(mcmc_results[col], [16, 50, 84])
                    minus_err = q50 - q16
                    plus_err = q84 - q50
                    param_stats.append({
                        'Parameter': col,
                        'Value': q50,
                        'Lower_Error': minus_err,
                        'Upper_Error': plus_err
                    })
                
                # Create DataFrame
                stats_df = pd.DataFrame(param_stats)
                
                print("\nParameter Statistics (median ± 1σ):")
                print(stats_df.to_string(index=False, float_format='%.4f'))
                
                # Try to create corner plot
                try:
                    print("\nGenerating corner plot...")
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    from scipy.stats import gaussian_kde
                    
                    with plot_area:
                        clear_output(wait=True)
                        
                        # Create corner plot first (above combined plot)
                        corner_fig = plotting_manager.create_corner_plot(mcmc_results, tool.current_galaxy.name, tool.config, fit_statistics=current_fit_stats, burn_in_length=burn_in, show_plot=True)
                        
                        
                        # Get data from modeling module  
                        profile_data = tool.get_radial_profile_data()
                        image_data = tool.get_image_data()
                        beta_params = tool.get_beta_model_params()
                        
                        # Check if we have fit results
                        try:
                            from sherpa.astro.ui import get_fit_results
                            get_fit_results()
                            fitted = True
                        except:
                            fitted = False
                        
                        # Get scale for kpc conversion (pc/arcsec)
                        pc_per_arcsec_scale = scale_input.value if scale_input.value > 0 else None
                        
                        # Then create comprehensive plot below it
                        fig = plotting_manager.create_comprehensive_plot(
                            profile_data, image_data, tool.current_galaxy.name,
                            tool.current_model.model_type if tool.current_model else "",
                            fitted, beta_params, distance_scale=pc_per_arcsec_scale
                        )
                        plt.show()
                        
                except ImportError:
                    print("\nNote: Install 'seaborn' package to generate corner plots")
                except Exception as e:
                    print(f"\nWarning: Could not create corner plot with seaborn: {e}")
                
                print(f"\nMCMC analysis complete. Check the output files for detailed results.")
                
            except Exception as e:
                status.value = f"<b>Status:</b> ❌ MCMC Error: {str(e)[:30]}..."
    
    def on_save_corner_plot(b):
        """Save the current corner plot to file"""
        with output_area:
            try:
                if current_mcmc_results is None:
                    print("No MCMC results available! Run MCMC first.")
                    return
                
                if tool.current_galaxy is None:
                    print("No galaxy loaded!")
                    return
                
                print("Saving corner plot...")
                
                # Create corner plot without displaying it
                # We need to get the burn-in value that was used during MCMC
                burn_in_used = mcmc_burnin_input.value if 'mcmc_burnin_input' in locals() else 0
                corner_fig = plotting_manager.create_corner_plot(current_mcmc_results, tool.current_galaxy.name, tool.config, fit_statistics=current_fit_stats, burn_in_length=burn_in_used, show_plot=False)
                
                # Create output folder based on galaxy name (without .fits)
                galaxy_name = tool.current_galaxy.name
                folder_name = galaxy_name.replace('.fits', '')
                import os
                os.makedirs(folder_name, exist_ok=True)
                
                # Create filename in the folder
                model_type = tool.current_model.model_type
                filename = os.path.join(folder_name, f"{galaxy_name}_{model_type.replace(' ', '_')}_corner_plot.png")
                
                # Save the figure
                corner_fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(corner_fig)
                
                print(f"Corner plot saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving corner plot: {e}")
    
    def on_save_combined_plot(b):
        """Save the current comprehensive plot (radial, residual, data, model) to file"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No model or galaxy loaded!")
                    return
                
                print("Saving comprehensive plot...")
                
                # Get data from modeling module
                profile_data = tool.get_radial_profile_data()
                image_data = tool.get_image_data()
                beta_params = tool.get_beta_model_params()
                
                # Check if we have fit results
                try:
                    from sherpa.astro.ui import get_fit_results
                    get_fit_results()
                    fitted = True
                except:
                    fitted = False
                
                # Get scale for kpc conversion (pc/arcsec)
                pc_per_arcsec_scale = scale_input.value if scale_input.value > 0 else None
                
                # Temporarily disable interactive mode to prevent display
                import matplotlib
                was_interactive = matplotlib.is_interactive()
                matplotlib.pyplot.ioff()
                
                try:
                    # Create comprehensive plot without showing it
                    fig = plotting_manager.create_comprehensive_plot(
                        profile_data, image_data, tool.current_galaxy.name,
                        tool.current_model.model_type if tool.current_model else "",
                        fitted, beta_params, distance_scale=pc_per_arcsec_scale
                    )
                    
                    # Create output folder based on galaxy name (without .fits)
                    galaxy_name = tool.current_galaxy.name
                    folder_name = galaxy_name.replace('.fits', '')
                    import os
                    os.makedirs(folder_name, exist_ok=True)
                    
                    # Create filename in the folder
                    model_type = tool.current_model.model_type
                    filename = os.path.join(folder_name, f"{galaxy_name}_{model_type.replace(' ', '_')}_combined_plot.png")
                    
                    # Save the figure
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"Combined plot saved to: {filename}")
                    
                finally:
                    # Restore previous interactive mode
                    if was_interactive:
                        matplotlib.pyplot.ion()
                
            except Exception as e:
                print(f"Error saving combined plot: {e}")
    
    # Connect event handlers
    fit_button.on_click(on_fit_model)
    mcmc_button.on_click(on_run_mcmc)
    save_params_button.on_click(on_save_parameters)
    save_residual_button.on_click(on_save_residual)
    save_model_button.on_click(on_save_model)
    save_combined_button.on_click(on_save_combined_plot)
    model_dropdown.observe(on_model_change, names='value')
    save_corner_button.on_click(on_save_corner_plot)
    galaxy_dropdown.observe(on_galaxy_change, names='value')
    size_dropdown.observe(on_scale_dropdown_change, names='value')
    scale_input.observe(on_scale_change, names='value')
    
    # First row of buttons: Fit Model, Save Params, Save Combined
    buttons_row1 = widgets.HBox([
        fit_button,
        save_params_button,
        save_combined_button
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        align_items='flex-start',
        padding='5px 0px'
    ))
    
    # Second row of buttons: Run MCMC, Save Corner, Save Model, Save Resid
    buttons_row2 = widgets.HBox([
        mcmc_button,
        save_corner_button,
        save_model_button,
        save_residual_button
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        align_items='flex-start',
        padding='5px 0px'
    ))
    
    # Model settings section (left column)
    model_settings = widgets.VBox([
        widgets.HTML("<h4 style='margin-bottom: 5px;'>Model Settings</h4>"),
        model_dropdown,
        method_dropdown,
        statistic_dropdown
    ], layout=widgets.Layout(
        width='49%',
        padding='1px'
    ))
    
    # MCMC settings section (right column)
    mcmc_settings = widgets.VBox([
        widgets.HTML("<h4 style='margin-bottom: 5px;'>MCMC Settings</h4>"),
        mcmc_length_input,
        mcmc_burnin_input
    ], layout=widgets.Layout(
        width='49%',
        padding='1px'
    ))
    
    # Model and MCMC settings side by side
    settings_row = widgets.HBox([
        model_settings,
        mcmc_settings
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start',
        margin='0px 0px 10px 0px'  # Add bottom margin for spacing
    ))
    
    # Layout: Main menu on left, plots on right, output below
    main_menu = widgets.VBox([
        widgets.HTML("<h4 style='margin-bottom: 5px;'>Image Settings</h4>"),
        galaxy_dropdown,
        size_dropdown,
        scale_input,
        settings_row,
        buttons_row1,
        buttons_row2,
        sliders_area
    ], layout=widgets.Layout(
        width='25%', 
        min_width='300px',
        height='100%',
        padding='5px', 
        overflow='auto',
        border='1px solid #ddd'
    ))
    


    
    # Plot area strictly constrained to right side
    plot_container = widgets.VBox([
        widgets.HTML("<h3>Plots</h3>"),
        plot_area
    ], layout=widgets.Layout(
        width='50%',
        height='100%',
        padding='25px',
        overflow='hidden',
        border='1px solid #ddd'
    ))
    
    # Output area below with same width
    output_container = widgets.VBox([
        widgets.HTML("<h3>Output</h3>"),
        output_area
    ], layout=widgets.Layout(
        width='25%',
        height='100%',
        padding='5px',
        overflow='scroll',
        border='1px solid #ddd'
    ))
    
    # Combine with strict side-by-side layout
    interface = widgets.HBox([
        main_menu,
        plot_container,
        output_container
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        align_items='flex-start'
    ))

    # Auto-load first galaxy
    def auto_load_first_galaxy():
        with output_area:
            clear_output()
            try:
                if tool.galaxy_files:
                    first_galaxy = next((f for f in tool.galaxy_files if f.name == 'NGC4649.fits'), tool.galaxy_files[0])
                    galaxy_dropdown.value = first_galaxy.name
                    print(f"Auto-loading: {first_galaxy.name}")
                    tool.load_galaxy(first_galaxy, scale=size_dropdown.value)
                    status.value = f"<b>Status:</b> ✅ Loaded: {tool.current_galaxy.name}"
                    
                    # Update size dropdown options based on loaded galaxy size
                    update_size_dropdown_options()
                    
                    # Auto-setup Single beta model
                    print("Auto-setting up Single beta model...")
                    tool.setup_model("Single beta")
                    print("Model setup completed!")
                    status.value = f"<b>Status:</b> ✅ Model: Single beta ready"
                    
                    # Create parameter sliders
                    create_parameter_sliders()
                    
                    # Sync sliders with backend state
                    update_sliders_from_backend()
                    
                    # Show initial frozen status
                    param_info = tool.get_parameter_info()
                    frozen_params = [p for p, info in param_info.items() if info['frozen']]
                    if frozen_params:
                        print(f"\nInitially frozen parameters: {', '.join(frozen_params)}")
                    
                    # Create comprehensive plot showing unfitted model
                    update_plot_with_current_model()
                else:
                    print("No galaxy files found!")
                    status.value = "<b>Status:</b> ❌ No galaxy files found"
                        
            except Exception as e:
                status.value = f"<b>Status:</b> ❌ Error: {str(e)[:50]}..."
                print(f"Error: {e}")
    
    # Trigger auto-load
    auto_load_first_galaxy()
    
    return interface, tool
