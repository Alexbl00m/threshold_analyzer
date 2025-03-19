import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit, minimize


def calculate_modified_dmax(intensity_values, lactate_values, baseline_lactate=0.8):
    """
    Modified Dmax method for threshold determination.
    
    This method:
    1. Finds the first point where lactate is 0.5 mmol/L above baseline
    2. Draws a line from this point to the max lactate point
    3. Finds the point on the lactate curve with maximum perpendicular distance from this line
    
    Args:
        intensity_values: Array of power or speed values
        lactate_values: Array of lactate values
        baseline_lactate: Resting lactate value
        
    Returns:
        threshold: The threshold intensity value
        details: Dictionary with additional analysis details
    """
    # Create a more densely sampled interpolation for better accuracy
    # First, ensure the data is sorted by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create a spline interpolation of the lactate curve
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Find the first point where lactate is 0.5 mmol/L above baseline
    threshold_mask = dense_lactate >= (baseline_lactate + 0.5)
    if not np.any(threshold_mask):
        # If no points meet criteria, return the highest intensity
        return intensity_sorted[-1], {
            "method": "Modified Dmax (fallback to max)", 
            "curve_fit": f,
            "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
        }
    
    first_point_idx = np.min(np.where(threshold_mask)[0])
    first_point = (dense_intensity[first_point_idx], dense_lactate[first_point_idx])
    
    # Maximum lactate point
    max_lactate_idx = np.argmax(dense_lactate)
    max_point = (dense_intensity[max_lactate_idx], dense_lactate[max_lactate_idx])
    
    # Calculate the perpendicular distance from each point to the line
    # Line equation: (y - y1) = m(x - x1)
    if max_point[0] - first_point[0] != 0:  # Avoid division by zero
        m = (max_point[1] - first_point[1]) / (max_point[0] - first_point[0])
        distances = np.abs(dense_lactate - first_point[1] - m * (dense_intensity - first_point[0])) / np.sqrt(1 + m**2)
    else:
        # Vertical line case (unlikely but handled)
        distances = np.abs(dense_intensity - first_point[0])
    
    # Points to consider: between first_point and max_point
    valid_indices = (dense_intensity >= first_point[0]) & (dense_intensity <= max_point[0])
    if np.any(valid_indices):
        distances[~valid_indices] = 0
        max_distance_idx = np.argmax(distances)
        threshold = dense_intensity[max_distance_idx]
    else:
        # Fallback if no valid points
        threshold = intensity_sorted[-1]
    
    # Calculate heart rate at threshold (if we had HR data)
    hr_at_threshold = None  # Would need HR data interpolation
    
    details = {
        "method": "Modified Dmax",
        "first_point": first_point,
        "max_point": max_point,
        "curve_fit": f,
        "hr_at_threshold": hr_at_threshold,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return threshold, details


def calculate_lactate_turnpoint(intensity_values, lactate_values):
    """
    Calculates the lactate turnpoint by finding the point of maximum curvature
    on the lactate curve. This is the point where the rate of increase of lactate
    begins to accelerate significantly.
    
    Args:
        intensity_values: Array of power or speed values
        lactate_values: Array of lactate values
        
    Returns:
        threshold: The threshold intensity value
        details: Dictionary with additional analysis details
    """
    # Sort values by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create a spline interpolation for smoother curve
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate first and second derivatives
    # For numerical stability, use central difference method
    h = dense_intensity[1] - dense_intensity[0]
    first_deriv = np.gradient(dense_lactate, h)
    second_deriv = np.gradient(first_deriv, h)
    
    # Calculate curvature: κ = |f''| / (1 + (f')²)^(3/2)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**(3/2)
    
    # Find the intensity at maximum curvature after the initial flat region
    # Skip the first ~20% of points to avoid early curve artifacts
    skip_points = int(0.2 * len(dense_intensity))
    max_curve_idx = skip_points + np.argmax(curvature[skip_points:])
    threshold = dense_intensity[max_curve_idx]
    
    details = {
        "method": "Lactate Turnpoint",
        "curve_fit": f,
        "hr_at_threshold": None  # Would require HR data
    }
    
    return threshold, details


def calculate_fixed_threshold(intensity_values, lactate_values, threshold_value=4.0):
    """
    Calculates the intensity at a fixed lactate threshold (commonly 4 mmol/L).
    
    Args:
        intensity_values: Array of power or speed values
        lactate_values: Array of lactate values
        threshold_value: The lactate concentration threshold (default 4.0 mmol/L)
        
    Returns:
        threshold: The threshold intensity value
        details: Dictionary with additional analysis details
    """
    # Sort values by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create a spline interpolation
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Find where lactate = threshold_value
    lactate_diff = np.abs(dense_lactate - threshold_value)
    threshold_idx = np.argmin(lactate_diff)
    threshold = dense_intensity[threshold_idx]
    
    # If the minimum lactate is higher than threshold or max is lower, indicate this
    method_name = f"{threshold_value} mmol/L Fixed Threshold"
    if np.min(lactate_sorted) > threshold_value:
        method_name += " (extrapolated below data)"
    elif np.max(lactate_sorted) < threshold_value:
        method_name += " (extrapolated above data)"
    
    details = {
        "method": method_name,
        "threshold_value": threshold_value,
        "curve_fit": f,
        "hr_at_threshold": None  # Would require HR data
    }
    
    return threshold, details


def calculate_individual_anaerobic_threshold(intensity_values, lactate_values, baseline_lactate=0.8):
    """
    Individual Anaerobic Threshold (IAT) calculation using the method where threshold
    is defined as the intensity where lactate increases by a fixed amount (often 0.5-1.5 mmol/L)
    above baseline.
    
    Args:
        intensity_values: Array of power or speed values
        lactate_values: Array of lactate values
        baseline_lactate: Resting lactate value
        
    Returns:
        threshold: The threshold intensity value
        details: Dictionary with additional analysis details
    """
    # Sort values by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create a spline interpolation
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate the intensity where lactate is baseline + delta (0.5-1.5 mmol/L, typically 1.0)
    delta = 1.0  # This could be a parameter
    iat_lactate = baseline_lactate + delta
    
    lactate_diff = np.abs(dense_lactate - iat_lactate)
    threshold_idx = np.argmin(lactate_diff)
    threshold = dense_intensity[threshold_idx]
    
    details = {
        "method": f"Individual Anaerobic Threshold (+{delta} mmol/L)",
        "baseline_lactate": baseline_lactate,
        "delta": delta,
        "curve_fit": f,
        "hr_at_threshold": None  # Would require HR data
    }
    
    return threshold, details


def critical_power_model(t, cp, w_prime):
    """3-parameter critical power model"""
    return cp + w_prime / t


def calculate_critical_power(intensity_values, lactate_values):
    """
    Estimates Critical Power (CP) using relationship between power and lactate.
    This is a simplified estimation - true CP is best measured with time-to-exhaustion tests.
    
    This method:
    1. Fits the lactate-power curve
    2. Identifies the power where the lactate curve starts to steepen dramatically
    3. Uses this as an approximation of CP
    
    Args:
        intensity_values: Array of power values
        lactate_values: Array of lactate values
        
    Returns:
        threshold: The critical power estimate
        details: Dictionary with additional analysis details
    """
    # Sort values by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create a spline interpolation
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate the first derivative of the lactate curve
    h = dense_intensity[1] - dense_intensity[0]
    first_deriv = np.gradient(dense_lactate, h)
    
    # Find the point where the rate of increase begins to accelerate significantly
    # Typically near where the slope reaches ~0.035-0.05 mmol/L/W
    target_slope = 0.04  # This could be a parameter
    slope_diff = np.abs(first_deriv - target_slope)
    cp_idx = np.argmin(slope_diff)
    cp = dense_intensity[cp_idx]
    
    # Alternative method: approximate as 90-95% of the power at 4 mmol/L
    _, fixed_threshold_details = calculate_fixed_threshold(intensity_values, lactate_values, 4.0)
    power_at_4mmol = fixed_threshold_details.get("threshold", None)
    
    # Use the more conservative of the two estimates
    if power_at_4mmol is not None:
        cp_alt = 0.92 * power_at_4mmol  # 92% of power at 4 mmol/L
        cp = min(cp, cp_alt)
    
    details = {
        "method": "Critical Power (estimated)",
        "curve_fit": f,
        "hr_at_threshold": None  # Would require HR data
    }
    
    return cp, details


def estimate_heart_rate_at_threshold(intensity_values, lactate_values, heart_rate_values, threshold_intensity):
    """
    Estimates the heart rate at a given threshold intensity by interpolation.
    
    Args:
        intensity_values: Array of power or speed values
        heart_rate_values: Array of heart rate values
        threshold_intensity: The threshold intensity value
        
    Returns:
        hr_at_threshold: Heart rate at threshold intensity
    """
    if heart_rate_values is None or len(heart_rate_values) != len(intensity_values):
        return None
    
    # Sort by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    hr_sorted = heart_rate_values[sorted_indices]
    
    # Interpolate heart rate
    hr_interp = interpolate.interp1d(
        intensity_sorted, 
        hr_sorted, 
        kind='linear', 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    hr_at_threshold = float(hr_interp(threshold_intensity))
    
    return hr_at_threshold
