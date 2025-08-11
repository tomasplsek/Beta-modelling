#!/usr/bin/env python3
"""
Utility Functions Module
========================

Common utility functions shared across the beta fitting modules.

Author: Beta-modelling project
"""

import warnings

def suppress_astropy_warnings():
    """Suppress common astropy warnings that clutter output."""
    # Disable astropy warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
    warnings.filterwarnings('ignore', module='astropy')
    from astropy.utils.exceptions import AstropyWarning
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings('ignore', category=AstropyWarning)
    warnings.filterwarnings('ignore', category=FITSFixedWarning)


def calculate_pixel_scale_from_wcs(wcs, default_scale: float = 0.492) -> float:
    """Calculate pixel scale in arcseconds per pixel from WCS.
    
    Args:
        wcs: Astropy WCS object (can be None)
        default_scale: Default pixel scale to use if WCS calculation fails
        
    Returns:
        Pixel scale in arcseconds per pixel
    """
    if wcs is None:
        return default_scale
    
    try:
        # Get pixel scale from WCS in degrees
        pixel_scale_deg = abs(wcs.wcs.cdelt[0])  # degrees per pixel
        pixel_scale_arcsec = pixel_scale_deg * 3600.0  # convert to arcsec per pixel
        return pixel_scale_arcsec
    except Exception:
        # Fallback if WCS calculation fails
        return default_scale
