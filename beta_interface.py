#!/usr/bin/env python3
"""
Beta Profile Interface Module
=============================

This module provides the interactive ipywidgets interface that coordinates
functionality between the modeling and plotting modules.

Author: Extracted from beta_fitting_improved.py
"""

import warnings
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Disable astropy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
warnings.filterwarnings('ignore', module='astropy')
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', category=FITSFixedWarning)

# Jupyter/IPython widgets
import ipywidgets as widgets
from IPython.display import clear_output, display
from ipywidgets import Layout, HBox, VBox

# Import from our modules
from beta_modeling import BetaFittingTool, FittingConfig
from beta_plotting import PlottingManager


def create_beta_fitting_interface():
    """Create an interactive widget interface for beta fitting with main menu layout."""
    tool = BetaFittingTool()
    plotting_manager = PlottingManager(tool.config)
    
    # Global variable to store parameter sliders
    param_sliders = {}
    
    # Output areas
    output_area = widgets.Output()
    plot_area = widgets.Output()
    sliders_area = widgets.Output()
    
    # Main menu widgets
    galaxy_dropdown = widgets.Dropdown(
        options=[f.name for f in tool.galaxy_files],
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
    
    scale_dropdown = widgets.Dropdown(
        options=tool.config.available_scales,
        value="original",
        description="Scale:  ",
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
    
    # Buttons
    fit_button = widgets.Button(description="Fit Model", button_style="success", layout=widgets.Layout(width='100px'))
    save_params_button = widgets.Button(description="Save Parameters", button_style="info", layout=widgets.Layout(width='120px'))
    save_residual_button = widgets.Button(description="Save Residual", button_style="warning", layout=widgets.Layout(width='120px'))
    save_model_button = widgets.Button(description="Save Model", button_style="primary", layout=widgets.Layout(width='120px'))
    
    # Status display
    status = widgets.HTML(value="<b>Status:</b> Ready")
    
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
                        layout=widgets.Layout(width='60%')
                    )
                else:
                    # Use regular slider for other parameters
                    slider = widgets.FloatSlider(
                        value=info['val'],
                        min=info['min'],
                        max=info['max'],
                        step=(info['max'] - info['min']) / 100,
                        description=display_name,
                        layout=widgets.Layout(width='60%')
                    )
                
                # Create completely independent freeze checkbox for this specific parameter
                freeze_checkbox = widgets.Checkbox(
                    value=info['frozen'],
                    description="Freeze",
                    layout=widgets.Layout(width='20%'),
                    tooltip=f'Freeze/unfreeze {param_name}',
                    indent=False
                )
                
                # Connect slider to update function with proper closure capture
                slider.observe(lambda change, pname=param_name: (
                    tool.update_parameter(pname, change['new']),
                    update_plot_with_current_model()
                ) if tool.current_model is not None else None, names='value')
                
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
                print("Model Parameters:")
                for param_name, info in param_info.items():
                    # Create widgets for this parameter
                    slider, freeze_checkbox = create_parameter_widgets(param_name, info)
                    
                    # Store widgets
                    param_sliders[param_name] = {
                        'slider': slider, 
                        'checkbox': freeze_checkbox
                    }
                    
                    # Create horizontal layout with slider and checkbox
                    param_row = widgets.HBox([slider, freeze_checkbox], 
                                           layout=widgets.Layout(width='100%', justify_content='space-between'))
                    display(param_row)

    def update_sliders_from_backend():
        """Update slider values and checkbox states from the backend parameter values."""
        if tool.current_model is None:
            return
        
        try:
            param_info = tool.get_parameter_info()
            for param_name, info in param_info.items():
                if param_name in param_sliders:
                    # Update slider value
                    param_sliders[param_name]['slider'].value = info['val']
                    # Update freeze checkbox
                    param_sliders[param_name]['checkbox'].value = info['frozen']
        except Exception as e:
            print(f"Error updating sliders from backend: {e}")
    
    def update_plot_with_current_model():
        """Update plot with current model parameters."""
        try:
            with plot_area:
                clear_output(wait=True)
                
                # Get data from modeling module
                profile_data = tool.get_radial_profile_data()
                image_data = tool.get_image_data()
                beta_params = tool.get_beta_model_params()
                
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
                print("Fitting completed!")
                
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
        if tool.current_galaxy is not None:
            with output_area:
                clear_output()
                try:
                    print(f"Changing to {change['new']} model...")
                    tool.setup_model(change['new'])
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
                tool.load_galaxy(galaxy_file, scale=scale_dropdown.value)
                status.value = f"<b>Status:</b> ✅ Loaded: {tool.current_galaxy.name}"
                
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
                
                # Create output filename
                filename = f"{galaxy_name}_{model_type.replace(' ', '_')}_params.txt"
                
                # Write parameters to file
                with open(filename, 'w') as f:
                    f.write(f"Beta Profile Fit Parameters\n")
                    f.write(f"Galaxy: {galaxy_name}\n")
                    f.write(f"Model: {model_type}\n")
                    f.write(f"Scale: {tool.current_scale}\n")
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
        """Save residual image with original header"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No fitted model or galaxy loaded!")
                    return
                
                from sherpa.astro.ui import get_resid_image, get_data
                from astropy.io import fits
                
                # Get residual data
                resid = get_resid_image().y.reshape(get_data().shape)
                
                # Copy header from original data
                original_header = tool.current_galaxy.header.copy()
                
                # Create filename
                galaxy_name = tool.current_galaxy.name
                model_type = tool.current_model.model_type
                filename = f"{galaxy_name}_{model_type.replace(' ', '_')}_residual.fits"
                
                # Save FITS file with copied header
                hdu = fits.PrimaryHDU(data=resid, header=original_header)
                hdu.writeto(filename, overwrite=True)
                
                print(f"Residual image saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving residual image: {e}")
    
    def on_save_model(b):
        """Save model image with original header"""
        with output_area:
            try:
                if tool.current_model is None or tool.current_galaxy is None:
                    print("No model or galaxy loaded!")
                    return
                
                from sherpa.astro.ui import get_model_image, get_data
                from astropy.io import fits
                
                # Get model data
                model_data = get_model_image().y.reshape(get_data().shape)
                
                # Copy header from original data
                original_header = tool.current_galaxy.header.copy()
                
                # Create filename
                galaxy_name = tool.current_galaxy.name
                model_type = tool.current_model.model_type
                filename = f"{galaxy_name}_{model_type.replace(' ', '_')}_model.fits"
                
                # Save FITS file with copied header
                hdu = fits.PrimaryHDU(data=model_data, header=original_header)
                hdu.writeto(filename, overwrite=True)
                
                print(f"Model image saved to: {filename}")
                
            except Exception as e:
                print(f"Error saving model image: {e}")
    
    def on_scale_change(change):
        """Handle scale input changes"""
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
    
    # Connect event handlers
    fit_button.on_click(on_fit_model)
    save_params_button.on_click(on_save_parameters)
    save_residual_button.on_click(on_save_residual)
    save_model_button.on_click(on_save_model)
    model_dropdown.observe(on_model_change, names='value')
    galaxy_dropdown.observe(on_galaxy_change, names='value')
    scale_input.observe(on_scale_change, names='value')
    
  # Combine with strict side-by-side layout
    buttons = widgets.HBox([
        fit_button,
        save_params_button,
        save_residual_button,
        save_model_button
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        align_items='flex-start'
    ))


    # Layout: Main menu on left, plots on right, output below
    main_menu = widgets.VBox([
        widgets.HTML("<h4>Image Settings</h4>"),
        galaxy_dropdown,
        scale_dropdown,
        scale_input,
        widgets.HTML("<h4>Model Settings</h4>"),
        model_dropdown,
        method_dropdown,
        statistic_dropdown,
        buttons,
        status,
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
                    first_galaxy = tool.galaxy_files[0]
                    galaxy_dropdown.value = first_galaxy.name
                    print(f"Auto-loading: {first_galaxy.name}")
                    tool.load_galaxy(first_galaxy, scale=scale_dropdown.value)
                    status.value = f"<b>Status:</b> ✅ Loaded: {tool.current_galaxy.name}"
                    
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
