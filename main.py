import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from scipy import interpolate
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Set page configuration
st.set_page_config(
    page_title="Threshold Analyzer - Lindblom Coaching",
    page_icon="🏊‍♂️🚴‍♂️🏃‍♂️",
    layout="wide"
)

# Define brand colors
BRAND_COLORS = {
    'primary': '#E6754E',    # Orange
    'secondary': '#2E4057',  # Dark blue
    'accent1': '#48A9A6',    # Teal
    'accent2': '#D4B483',    # Light brown
    'accent3': '#C1666B',    # Red
    'background': '#F8F8F8', # Light gray
    'text': '#333333'        # Dark gray
}

THRESHOLD_COLORS = {
    'Modified Dmax': BRAND_COLORS['primary'],
    'Lactate Turnpoint': BRAND_COLORS['accent1'],
    '4 mmol/L Fixed Threshold': BRAND_COLORS['accent3'],
    'Individual Anaerobic Threshold': BRAND_COLORS['secondary'],
    'Critical Power': BRAND_COLORS['accent2']
}

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

.main {
    background-color: #FFFFFF;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    color: #E6754E;
}

.stButton>button {
    background-color: #E6754E;
    color: white;
    font-family: 'Montserrat', sans-serif;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

.stButton>button:hover {
    background-color: #c45d3a;
}

.highlight {
    color: #E6754E;
    font-weight: 600;
}

footer {
    font-family: 'Montserrat', sans-serif;
    font-size: 12px;
    color: #888888;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

#------ Utility Functions ------#

def calculate_effective_intensity(intensity, step_duration, standard_duration=5):
    """Calculate effective intensity for incomplete steps."""
    if step_duration >= standard_duration:
        return intensity
    
    completion_percentage = step_duration / standard_duration
    adjustment_factor = 1 + max(0, (1 - completion_percentage)) * 0.15
    return intensity * adjustment_factor

def pace_to_speed(pace):
    """Convert pace (min/km) to speed (km/h)."""
    try:
        minutes, seconds = map(int, pace.split(':'))
        total_seconds = minutes * 60 + seconds
        return 3600 / total_seconds
    except:
        return 0

def speed_to_pace(speed):
    """Convert speed (km/h) to pace (min/km)."""
    try:
        if speed <= 0:
            return "0:00"
        minutes_per_km = 60 / speed
        minutes = int(minutes_per_km)
        seconds = int((minutes_per_km - minutes) * 60)
        return f"{minutes}:{seconds:02d}"
    except:
        return "0:00"

def calculate_training_zones(threshold_value, sport, max_hr=None, threshold_hr=None):
    """Calculate training zones based on threshold."""
    zones = []
    
    if sport == "Cycling":
        # Cycling zones based on percentage of threshold power
        zones = [
            {
                "Zone": "Zone 1 - Recovery",
                "Power Range": f"<{int(threshold_value * 0.55)} W",
                "Percentage of FTP": "<55%",
                "Description": "Very easy, active recovery"
            },
            {
                "Zone": "Zone 2 - Endurance",
                "Power Range": f"{int(threshold_value * 0.55)}-{int(threshold_value * 0.75)} W",
                "Percentage of FTP": "55-75%",
                "Description": "All day pace, fat burning, endurance building"
            },
            {
                "Zone": "Zone 3 - Tempo",
                "Power Range": f"{int(threshold_value * 0.75)}-{int(threshold_value * 0.90)} W",
                "Percentage of FTP": "75-90%",
                "Description": "Moderate intensity, improved efficiency"
            },
            {
                "Zone": "Zone 4 - Threshold",
                "Power Range": f"{int(threshold_value * 0.90)}-{int(threshold_value * 1.05)} W",
                "Percentage of FTP": "90-105%",
                "Description": "Lactate threshold, race pace for time trials"
            },
            {
                "Zone": "Zone 5 - VO2max",
                "Power Range": f"{int(threshold_value * 1.05)}-{int(threshold_value * 1.20)} W",
                "Percentage of FTP": "105-120%",
                "Description": "Maximum oxygen uptake, high intensity intervals"
            },
            {
                "Zone": "Zone 6 - Anaerobic",
                "Power Range": f"{int(threshold_value * 1.20)}-{int(threshold_value * 1.50)} W",
                "Percentage of FTP": "120-150%",
                "Description": "Short, intense efforts, sprint training"
            },
            {
                "Zone": "Zone 7 - Neuromuscular",
                "Power Range": f">{int(threshold_value * 1.50)} W",
                "Percentage of FTP": ">150%",
                "Description": "Max power, short sprints, peak power"
            }
        ]
        
        # Add heart rate if available
        if threshold_hr is not None:
            for zone in zones:
                zone_name = zone["Zone"]
                if "Zone 1" in zone_name:
                    zone["Heart Rate"] = f"<{int(threshold_hr * 0.82)} bpm"
                elif "Zone 2" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.82)}-{int(threshold_hr * 0.89)} bpm"
                elif "Zone 3" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.89)}-{int(threshold_hr * 0.94)} bpm"
                elif "Zone 4" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.94)}-{int(threshold_hr * 1.00)} bpm"
                elif "Zone 5" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 1.00)}-{int(threshold_hr * 1.03)} bpm"
                elif "Zone 6" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 1.03)}-{int(threshold_hr * 1.06)} bpm"
                elif "Zone 7" in zone_name:
                    zone["Heart Rate"] = f">{int(threshold_hr * 1.06)} bpm"
    else:  # Running
        # Convert threshold to pace
        threshold_pace_mins = int(60 / threshold_value)
        threshold_pace_secs = int(60 * (60 / threshold_value - threshold_pace_mins))
        
        # Running zones
        zones = [
            {
                "Zone": "Zone 1 - Recovery",
                "Speed Range": f"<{threshold_value * 0.70:.1f} km/h",
                "Pace Range": pace_range(threshold_value * 0.70, None, slower=True),
                "Percentage of Threshold": "<70%",
                "Description": "Very easy, recovery runs"
            },
            {
                "Zone": "Zone 2 - Endurance",
                "Speed Range": f"{threshold_value * 0.70:.1f}-{threshold_value * 0.80:.1f} km/h",
                "Pace Range": pace_range(threshold_value * 0.70, threshold_value * 0.80),
                "Percentage of Threshold": "70-80%",
                "Description": "Easy aerobic running, long runs"
            },
            {
                "Zone": "Zone 3 - Tempo",
                "Speed Range": f"{threshold_value * 0.80:.1f}-{threshold_value * 0.90:.1f} km/h",
                "Pace Range": pace_range(threshold_value * 0.80, threshold_value * 0.90),
                "Percentage of Threshold": "80-90%",
                "Description": "Steady state, marathon pace"
            },
            {
                "Zone": "Zone 4 - Threshold",
                "Speed Range": f"{threshold_value * 0.90:.1f}-{threshold_value * 1.05:.1f} km/h",
                "Pace Range": pace_range(threshold_value * 0.90, threshold_value * 1.05),
                "Percentage of Threshold": "90-105%",
                "Description": "Lactate threshold, comfortably hard"
            },
            {
                "Zone": "Zone 5 - VO2max",
                "Speed Range": f"{threshold_value * 1.05:.1f}-{threshold_value * 1.15:.1f} km/h",
                "Pace Range": pace_range(threshold_value * 1.05, threshold_value * 1.15),
                "Percentage of Threshold": "105-115%",
                "Description": "VO2max intervals, 5K pace"
            }
        ]
        
        # Add heart rate if available
        if threshold_hr is not None:
            for zone in zones:
                zone_name = zone["Zone"]
                if "Zone 1" in zone_name:
                    zone["Heart Rate"] = f"<{int(threshold_hr * 0.80)} bpm"
                elif "Zone 2" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.80)}-{int(threshold_hr * 0.87)} bpm"
                elif "Zone 3" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.87)}-{int(threshold_hr * 0.93)} bpm"
                elif "Zone 4" in zone_name:
                    zone["Heart Rate"] = f"{int(threshold_hr * 0.93)}-{int(threshold_hr * 1.00)} bpm"
                elif "Zone 5" in zone_name:
                    zone["Heart Rate"] = f">{int(threshold_hr * 1.00)} bpm"
    
    return zones

def pace_range(speed_low, speed_high, slower=False):
    """Convert speed range to pace range."""
    if speed_low <= 0:
        return "N/A"
    
    mins_low = int(60 / speed_low)
    secs_low = int(60 * (60 / speed_low - mins_low))
    pace_low = f"{mins_low}:{secs_low:02d}"
    
    if speed_high is not None:
        mins_high = int(60 / speed_high)
        secs_high = int(60 * (60 / speed_high - mins_high))
        pace_high = f"{mins_high}:{secs_high:02d}"
        
        if slower:
            return f"Slower than {pace_high}"
        else:
            return f"{pace_high} - {pace_low}"
    else:
        if slower:
            return f"Slower than {pace_low}"
        else:
            return f"Faster than {pace_low}"

def estimate_vo2max(threshold_value, weight, sport, gender="Male"):
    """Estimate VO2max based on threshold."""
    if sport == "Cycling":
        gender_factor = 1.0 if gender == "Male" else 0.9
        vo2max = (10.8 * threshold_value / weight + 7) * gender_factor
    else:
        gender_factor = 1.0 if gender == "Male" else 0.92
        speed_in_ms = threshold_value / 3.6
        vo2max = (4.5 * speed_in_ms + 3.5) * gender_factor
    
    return vo2max

#------ Threshold Calculation Models ------#

def calculate_modified_dmax(intensity_values, lactate_values, baseline_lactate=0.8):
    """Modified Dmax method."""
    # Sort data by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    # Create spline interpolation
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Find first point where lactate is 0.5 mmol/L above baseline
    threshold_mask = dense_lactate >= (baseline_lactate + 0.5)
    if not np.any(threshold_mask):
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
    
    # Calculate perpendicular distance
    if max_point[0] - first_point[0] != 0:
        m = (max_point[1] - first_point[1]) / (max_point[0] - first_point[0])
        distances = np.abs(dense_lactate - first_point[1] - m * (dense_intensity - first_point[0])) / np.sqrt(1 + m**2)
    else:
        distances = np.abs(dense_intensity - first_point[0])
    
    # Find max distance point
    valid_indices = (dense_intensity >= first_point[0]) & (dense_intensity <= max_point[0])
    if np.any(valid_indices):
        distances[~valid_indices] = 0
        max_distance_idx = np.argmax(distances)
        threshold = dense_intensity[max_distance_idx]
    else:
        threshold = intensity_sorted[-1]
    
    details = {
        "method": "Modified Dmax",
        "first_point": first_point,
        "max_point": max_point,
        "curve_fit": f,
        "hr_at_threshold": None,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return threshold, details

def calculate_lactate_turnpoint(intensity_values, lactate_values):
    """Lactate turnpoint method."""
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate derivatives
    h = dense_intensity[1] - dense_intensity[0]
    first_deriv = np.gradient(dense_lactate, h)
    second_deriv = np.gradient(first_deriv, h)
    
    # Calculate curvature
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**(3/2)
    
    # Find max curvature point
    skip_points = int(0.2 * len(dense_intensity))
    max_curve_idx = skip_points + np.argmax(curvature[skip_points:])
    threshold = dense_intensity[max_curve_idx]
    
    details = {
        "method": "Lactate Turnpoint",
        "curve_fit": f,
        "hr_at_threshold": None,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return threshold, details

def calculate_fixed_threshold(intensity_values, lactate_values, threshold_value=4.0):
    """Fixed lactate threshold method."""
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Find where lactate equals threshold value
    lactate_diff = np.abs(dense_lactate - threshold_value)
    threshold_idx = np.argmin(lactate_diff)
    threshold = dense_intensity[threshold_idx]
    
    method_name = f"{threshold_value} mmol/L Fixed Threshold"
    if np.min(lactate_sorted) > threshold_value:
        method_name += " (extrapolated below data)"
    elif np.max(lactate_sorted) < threshold_value:
        method_name += " (extrapolated above data)"
    
    details = {
        "method": method_name,
        "threshold_value": threshold_value,
        "curve_fit": f,
        "hr_at_threshold": None,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return threshold, details

def calculate_individual_anaerobic_threshold(intensity_values, lactate_values, baseline_lactate=0.8):
    """Individual anaerobic threshold method."""
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate IAT as baseline + delta
    delta = 1.0
    iat_lactate = baseline_lactate + delta
    
    lactate_diff = np.abs(dense_lactate - iat_lactate)
    threshold_idx = np.argmin(lactate_diff)
    threshold = dense_intensity[threshold_idx]
    
    details = {
        "method": f"Individual Anaerobic Threshold (+{delta} mmol/L)",
        "baseline_lactate": baseline_lactate,
        "delta": delta,
        "curve_fit": f,
        "hr_at_threshold": None,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return threshold, details

def calculate_critical_power(intensity_values, lactate_values):
    """Critical power estimation."""
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    lactate_sorted = lactate_values[sorted_indices]
    
    f = interpolate.interp1d(intensity_sorted, lactate_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    dense_intensity = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
    dense_lactate = f(dense_intensity)
    
    # Calculate first derivative
    h = dense_intensity[1] - dense_intensity[0]
    first_deriv = np.gradient(dense_lactate, h)
    
    # Find point with target slope
    target_slope = 0.04
    slope_diff = np.abs(first_deriv - target_slope)
    cp_idx = np.argmin(slope_diff)
    cp = dense_intensity[cp_idx]
    
    # Alternative estimate: 92% of power at 4 mmol/L
    _, fixed_threshold_details = calculate_fixed_threshold(intensity_values, lactate_values, 4.0)
    power_at_4mmol = fixed_threshold_details.get("threshold", None)
    
    if power_at_4mmol is not None:
        cp_alt = 0.92 * power_at_4mmol
        cp = min(cp, cp_alt)
    
    details = {
        "method": "Critical Power (estimated)",
        "curve_fit": f,
        "hr_at_threshold": None,
        "uses_effective_intensity": "effective" in str(intensity_values.dtype).lower() or "effective" in str(type(intensity_values)).lower()
    }
    
    return cp, details

def estimate_heart_rate_at_threshold(intensity_values, heart_rate_values, threshold_intensity):
    """Estimate heart rate at threshold intensity."""
    if heart_rate_values is None or len(heart_rate_values) != len(intensity_values):
        return None
    
    # Sort by intensity
    sorted_indices = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sorted_indices]
    hr_sorted = heart_rate_values[sorted_indices]
    
    hr_interp = interpolate.interp1d(
        intensity_sorted, 
        hr_sorted, 
        kind='linear', 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    hr_at_threshold = float(hr_interp(threshold_intensity))
    return hr_at_threshold

#------ Visualization Functions ------#

def create_lactate_curve_plot(intensity_values, lactate_values, heart_rate_values, results, x_label="Power (Watts)", sport="Cycling"):
    """Create matplotlib lactate curve plot."""
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Sort data
    sort_idx = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sort_idx]
    lactate_sorted = lactate_values[sort_idx]
    
    if heart_rate_values is not None and len(heart_rate_values) == len(intensity_values):
        hr_sorted = heart_rate_values[sort_idx]
    else:
        hr_sorted = np.zeros_like(intensity_sorted)
    
    # Plot lactate curve
    ax1.scatter(intensity_sorted, lactate_sorted, s=80, color=BRAND_COLORS['secondary'], label='Test Data')
    
    # Add spline fits
    for method_name, result in results.items():
        if 'curve_fit' in result['details']:
            curve_fit = result['details']['curve_fit']
            dense_x = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
            dense_y = curve_fit(dense_x)
            ax1.plot(dense_x, dense_y, '--', linewidth=1, alpha=0.7, 
                     color=THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text']))
    
    # Plot thresholds
    for i, (method_name, result) in enumerate(results.items()):
        threshold = result['threshold']
        color = THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text'])
        
        # Vertical line
        ax1.axvline(x=threshold, color=color, linestyle='--', alpha=0.7)
        
        # Plot threshold point
        if 'curve_fit' in result['details']:
            curve_fit = result['details']['curve_fit']
            lactate_at_threshold = curve_fit(threshold)
            ax1.scatter([threshold], [lactate_at_threshold], marker='o', s=100, 
                        color=color, edgecolor='white', zorder=10)
        
        # Label
        if sport == "Cycling":
            label = f"{method_name}: {threshold:.1f} W"
        else:
            pace_min = int(60 / threshold)
            pace_sec = int(60 * (60 / threshold - pace_min))
            label = f"{method_name}: {threshold:.1f} km/h ({pace_min}:{pace_sec:02d} min/km)"
        
        ax1.annotate(label, xy=(threshold, 0), xytext=(threshold, -0.5 - i*0.3),
                     arrowprops=dict(arrowstyle='->', color=color),
                     va='top', ha='center', color=color)
    
    # Lactate subplot settings
    ax1.set_title('Lactate Response Curve', fontsize=16, color=BRAND_COLORS['primary'])
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Lactate (mmol/L)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    # Plot heart rate
    ax2.scatter(intensity_sorted, hr_sorted, s=80, color=BRAND_COLORS['accent3'], label='Heart Rate')
    
    # Fit heart rate curve
    if len(intensity_sorted) >= 3:
        try:
            hr_fit = np.polyfit(intensity_sorted, hr_sorted, 1)
            hr_line = np.poly1d(hr_fit)
            dense_x = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
            ax2.plot(dense_x, hr_line(dense_x), '--', color=BRAND_COLORS['accent3'], alpha=0.7)
        except:
            pass
    
    # Plot thresholds on HR subplot
    for method_name, result in results.items():
        threshold = result['threshold']
        color = THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text'])
        
        ax2.axvline(x=threshold, color=color, linestyle='--', alpha=0.7)
        
        # Plot heart rate at threshold if we have the model
        if len(intensity_sorted) >= 3:
            try:
                hr_at_threshold = hr_line(threshold)
                ax2.scatter([threshold], [hr_at_threshold], marker='o', s=100, 
                            color=color, edgecolor='white', zorder=10)
                
                # Store HR in results
                results[method_name]['details']['hr_at_threshold'] = hr_at_threshold
            except:
                pass
    
    # Heart rate subplot settings
    ax2.set_title('Heart Rate Response', fontsize=16, color=BRAND_COLORS['primary'])
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    min_hr = max(60, np.min(hr_sorted) - 10)
    ax2.set_ylim(bottom=min_hr)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    fig.text(0.5, 0.01, "© Lindblom Coaching", ha='center', 
             color=BRAND_COLORS['secondary'], alpha=0.7, fontsize=8)
    
    return fig

# Continue create_interactive_plot function
# Continue create_interactive_plot function
def create_interactive_plot(data, results, x_column="Power", sport="Cycling"):
    """Create interactive Plotly plot."""
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Lactate Response Curve', 'Heart Rate Response')
    )
    
    # Add lactate data points
    fig.add_trace(
        go.Scatter(
            x=data[x_column], 
            y=data['Lactate'],
            mode='markers+lines',
            name='Lactate',
            marker=dict(size=12, color=BRAND_COLORS['secondary']),
            line=dict(dash='dot', width=1, color=BRAND_COLORS['secondary']),
        ),
        row=1, col=1
    )
    
    # Add heart rate data points
    if 'HeartRate' in data.columns and data['HeartRate'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=data[x_column], 
                y=data['HeartRate'],
                mode='markers+lines',
                name='Heart Rate',
                marker=dict(size=12, color=BRAND_COLORS['accent3']),
                line=dict(dash='dot', width=1, color=BRAND_COLORS['accent3']),
            ),
            row=2, col=1
        )
    
    # Add threshold lines and annotations for each method
    for method_name, result in results.items():
        threshold = result['threshold']
        color = THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text'])
        
        # Add vertical threshold lines
        fig.add_vline(
            x=threshold, 
            line_width=2, 
            line_dash="dash", 
            line_color=color,
            row="all", col=1
        )
        
        # Add annotation for the method name
        fig.add_annotation(
            x=threshold,
            y=1.05,
            text=method_name,
            showarrow=False,
            xref="x",
            yref="paper",
            font=dict(color=color, size=10),
            row=1, col=1
        )
        
        # Add threshold points on the lactate curve
        if 'curve_fit' in result['details'] and result['details']['curve_fit'] is not None:
            # Get lactate value at threshold
            try:
                curve_fit = result['details']['curve_fit']
                lactate_at_threshold = curve_fit(threshold)
                
                # Add point marker
                fig.add_trace(
                    go.Scatter(
                        x=[threshold],
                        y=[lactate_at_threshold],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=14,
                            color=color,
                            line=dict(color='white', width=2)
                        ),
                        name=f"{method_name} Threshold",
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add annotation with lactate value
                fig.add_annotation(
                    x=threshold,
                    y=lactate_at_threshold,
                    text=f"{lactate_at_threshold:.2f} mmol/L",
                    showarrow=False,
                    yshift=20,
                    font=dict(color=color, size=10),
                    row=1, col=1
                )
            except Exception:
                pass
        
        # Add threshold points on the heart rate curve
        if ('hr_at_threshold' in result['details'] and 
            result['details']['hr_at_threshold'] is not None and
            'HeartRate' in data.columns):
            
            hr_at_threshold = result['details']['hr_at_threshold']
            
            fig.add_trace(
                go.Scatter(
                    x=[threshold],
                    y=[hr_at_threshold],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=14,
                        color=color,
                        line=dict(color='white', width=2)
                    ),
                    name=f"{method_name} HR",
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add annotation with HR value
            fig.add_annotation(
                x=threshold,
                y=hr_at_threshold,
                text=f"{int(hr_at_threshold)} bpm",
                showarrow=False,
                yshift=20,
                font=dict(color=color, size=10),
                row=2, col=1
            )
    
    # Update layout
    x_title = "Power (Watts)" if sport == "Cycling" else "Speed (km/h)"
    fig.update_layout(
        title={
            'text': f"Lactate Threshold Analysis - {sport}",
            'font': {'size': 24, 'color': BRAND_COLORS['primary']},
            'y': 0.95
        },
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100)
    )
    
    # Update x and y axis labels and styling
    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(size=14, color=BRAND_COLORS['text']),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Lactate (mmol/L)",
        title_font=dict(size=14, color=BRAND_COLORS['text']),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Heart Rate (bpm)",
        title_font=dict(size=14, color=BRAND_COLORS['text']),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    # Add annotations for hover information
    fig.update_traces(hoverinfo="x+y")
    
    return fig

#------ PDF Report Generation ------#

def generate_pdf_report(athlete_info, data, results, training_zones, sport, 
                       additional_notes=None, include_logo=True, 
                       include_training_zones=True, include_raw_data=False):
    """Generate PDF report with threshold results."""
    buffer = io.BytesIO()
    
    # Set up the document
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles - use unique names to avoid conflicts
    if 'LCTitle' not in styles:
        styles.add(ParagraphStyle(
            name='LCTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor(BRAND_COLORS['primary']),
            alignment=TA_CENTER,
            spaceAfter=12
        ))
    
    if 'LCSubtitle' not in styles:
        styles.add(ParagraphStyle(
            name='LCSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor(BRAND_COLORS['secondary']),
            spaceAfter=10
        ))
    
    if 'LCSectionTitle' not in styles:
        styles.add(ParagraphStyle(
            name='LCSectionTitle',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor(BRAND_COLORS['primary']),
            spaceAfter=8
        ))
    
    if 'LCNormal' not in styles:
        styles.add(ParagraphStyle(
            name='LCNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor(BRAND_COLORS['text']),
            spaceAfter=6
        ))
    
    if 'LCCenterText' not in styles:
        styles.add(ParagraphStyle(
            name='LCCenterText',
            parent=styles['Normal'],
            alignment=TA_CENTER
        ))
    
    # Create flowable elements for the report
    story = []
    
    # Add title
    title = f"{sport} Threshold Analysis Report"
    story.append(Paragraph(title, styles['LCTitle']))
    
    # Add date
    date_str = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Report Date: {date_str}", styles['LCCenterText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Athlete information
    story.append(Paragraph("Athlete Information", styles['LCSectionTitle']))
    
    if athlete_info:
        # Extract athlete info
        name = athlete_info.get('name', 'N/A')
        gender = athlete_info.get('gender', 'N/A')
        weight = athlete_info.get('weight', 0)
        height = athlete_info.get('height', 0)
        
        # Create table for athlete info
        athlete_data = [
            ["Name:", name],
            ["Gender:", gender],
            ["Weight:", f"{weight} kg"],
            ["Height:", f"{height} cm"]
        ]
        
        t = Table(athlete_data, colWidths=[100, 300])
        t.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor(BRAND_COLORS['secondary'])),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Threshold Results
    story.append(Paragraph("Threshold Results", styles['LCSectionTitle']))
    
    # Table of threshold values
    if results:
        threshold_data = [["Method", "Threshold", "Relative"]]
        
        for method_name, result in results.items():
            threshold = result['threshold']
            
            # Check if this uses effective intensity
            uses_effective = result['details'].get('uses_effective_intensity', False)
            effective_label = " (Effective)" if uses_effective else ""
            
            if sport == "Cycling":
                threshold_str = f"{threshold:.1f} W{effective_label}"
                relative_str = f"{threshold / athlete_info.get('weight', 1):.2f} W/kg"
            else:
                threshold_str = f"{threshold:.2f} km/h{effective_label}"
                pace_min = int(60 / threshold)
                pace_sec = int((60 / threshold - pace_min) * 60)
                pace_str = f"{pace_min}:{pace_sec:02d} min/km"
                relative_str = pace_str
            
            threshold_data.append([method_name, threshold_str, relative_str])
        
        t = Table(threshold_data, colWidths=[180, 120, 120])
        t.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(BRAND_COLORS['secondary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Training Zones if requested
    if include_training_zones and training_zones:
        story.append(Paragraph("Training Zones", styles['LCSectionTitle']))
        
        # Column headers depend on sport
        if sport == "Cycling":
            zone_headers = ["Zone", "Power Range", "Heart Rate", "Description"]
            col_widths = [100, 100, 100, 150]
        else:
            zone_headers = ["Zone", "Speed Range", "Pace Range", "Heart Rate", "Description"]
            col_widths = [100, 80, 80, 80, 110]
        
        # Build table data
        zone_data = [zone_headers]
        
        for zone in training_zones:
            row = []
            for header in zone_headers:
                if header in zone:
                    row.append(zone[header])
                else:
                    row.append("")
            zone_data.append(row)
        
        # Create table
        t = Table(zone_data, colWidths=col_widths)
        
        # Apply styles
        table_style = [
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(BRAND_COLORS['secondary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        
        # Alternate row colors for better readability
        for i in range(1, len(zone_data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor(BRAND_COLORS['background'])))
        
        t.setStyle(TableStyle(table_style))
        story.append(t)
    
    # Additional notes if provided
    if additional_notes:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Additional Notes", styles['LCSectionTitle']))
        story.append(Paragraph(additional_notes, styles['LCNormal']))
    
    # Raw test data if requested
    if include_raw_data and not data.empty:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Raw Test Data", styles['LCSectionTitle']))
        
        # Use only relevant columns
        if sport == "Cycling":
            display_cols = ["Power", "HeartRate", "Lactate", "RPE", "StepDuration"]
        else:
            display_cols = ["Speed", "Pace", "HeartRate", "Lactate", "RPE", "StepDuration"]
        
        # Filter to keep only columns that exist in the data
        display_cols = [col for col in display_cols if col in data.columns]
        
        # Prepare data for table
        raw_data = [display_cols]  # Header row
        for _, row in data.iterrows():
            raw_data.append([str(row[col]) for col in display_cols])
        
        # Create table
        t = Table(raw_data)
        
        # Apply styles
        table_style = [
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(BRAND_COLORS['secondary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        
        # Alternate row colors
        for i in range(1, len(raw_data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor(BRAND_COLORS['background'])))
        
        t.setStyle(TableStyle(table_style))
        story.append(t)
    
    # Footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor(BRAND_COLORS['secondary']))
        footer_text = "© Lindblom Coaching - Professional Threshold Analysis"
        canvas.drawCentredString(doc.width / 2 + doc.leftMargin, 0.5 * inch, footer_text)
        canvas.restoreState()
    
    # Build document
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    
    # Get PDF as bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

#------ Data Processing ------#

def process_input_data(data, sport):
    """
    Process and prepare raw input data for analysis.
    
    This function handles:
    1. Setting default step durations if not provided
    2. Calculating effective intensity for incomplete steps
    3. Converting between pace and speed for running
    4. Formatting and organizing the data for analysis
    
    Args:
        data: DataFrame containing test data
        sport: Either "Cycling" or "Running"
        
    Returns:
        processed_data: Processed DataFrame ready for analysis
    """
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Set default step duration if not provided
    if 'StepDuration' not in processed_data.columns:
        default_duration = 5 if sport == "Cycling" else 4  # Default: 5 min for cycling, 4 min for running
        processed_data['StepDuration'] = default_duration
    
    # Sort by intensity
    intensity_col = "Power" if sport == "Cycling" else "Speed"
    processed_data = processed_data.sort_values(by=intensity_col)
    
    # Check for incomplete steps and calculate effective intensity
    standard_duration = 5 if sport == "Cycling" else 4
    
    # Create a new column for effective intensity if any steps are incomplete
    if (processed_data['StepDuration'] < standard_duration).any():
        # Calculate effective intensity for all steps
        processed_data[f'Effective{intensity_col}'] = processed_data.apply(
            lambda row: calculate_effective_intensity(
                row[intensity_col], 
                row['StepDuration'], 
                standard_duration
            ),
            axis=1
        )
        
        # Identify incomplete steps for display to user
        incomplete_steps = processed_data[processed_data['StepDuration'] < standard_duration]
        if not incomplete_steps.empty:
            intensity_unit = "W" if sport == "Cycling" else "km/h"
            for _, row in incomplete_steps.iterrows():
                # Calculate adjustment percentage for better understanding
                adjustment_pct = ((row[f'Effective{intensity_col}'] / row[intensity_col]) - 1) * 100
                st.info(f"Step at {row[intensity_col]} {intensity_unit} was adjusted to effective {row[f'Effective{intensity_col}']:.1f} {intensity_unit} " +
                       f"(+{adjustment_pct:.1f}%) based on {row['StepDuration']:.2f} min duration instead of standard {standard_duration} min.")
    
    # Handle missing HR data if needed
    if 'HeartRate' not in processed_data.columns:
        processed_data['HeartRate'] = None
    
    # Add pace calculation for running
    if sport == "Running" and 'Pace' not in processed_data.columns:
        processed_data['Pace'] = processed_data['Speed'].apply(
            lambda x: f"{int(60/x)}:{int((60/x - int(60/x))*60):02d}" if x > 0 else "0:00"
        )
        
        # Add effective pace if effective speed exists
        if 'EffectiveSpeed' in processed_data.columns:
            processed_data['EffectivePace'] = processed_data['EffectiveSpeed'].apply(
                lambda x: f"{int(60/x)}:{int((60/x - int(60/x))*60):02d}" if x > 0 else "0:00"
            )
    
    # Fill missing RPE values if needed
    if 'RPE' not in processed_data.columns:
        processed_data['RPE'] = None
    
    return processed_data

def validate_data(data, sport):
    """Validate that input data has required columns and structure."""
    # Check if DataFrame is empty
    if data.empty:
        return False, "Data is empty"
    
    # Check required columns based on sport
    if sport == "Cycling":
        required_columns = ["Power", "Lactate"]
    else:  # Running
        required_columns = ["Speed", "Lactate"]
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if there are enough data points (at least 3)
    if len(data) < 3:
        return False, "At least 3 data points are required for analysis"
    
    # Check if intensity values are positive
    intensity_col = "Power" if sport == "Cycling" else "Speed"
    if (data[intensity_col] <= 0).any():
        return False, f"{intensity_col} values must be positive"
    
    # Check if lactate values are non-negative
    if (data["Lactate"] < 0).any():
        return False, "Lactate values cannot be negative"
    
    # All checks passed
    return True, ""

#------ Main Application ------#

def data_upload_form(sport):
    """Display form for uploading CSV/Excel files."""
    uploaded_file = st.file_uploader("Upload test data file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                test_data = pd.read_csv(uploaded_file)
            else:
                test_data = pd.read_excel(uploaded_file)
            
            # Basic validation
            is_valid, message = validate_data(test_data, sport)
            if is_valid:
                st.success("Data successfully loaded!")
                
                # Check if StepDuration column exists, if not, allow user to add it
                if "StepDuration" not in test_data.columns:
                    if st.checkbox("Add step duration information", value=False):
                        st.info("Please specify the duration of each step. Default is 5 minutes for cycling, 4 minutes for running.")
                        
                        # Default step duration
                        default_duration = 5 if sport == "Cycling" else 4
                        
                        # Show a slider for standard step duration
                        standard_duration = st.slider("Standard step duration (minutes)", 1, 10, default_duration)
                        
                        # Checkbox for final step being incomplete
                        final_step_incomplete = st.checkbox("Was the final step incomplete?", value=False)
                        
                        if final_step_incomplete:
                            # If final step was incomplete, get the actual duration
                            col1, col2 = st.columns(2)
                            with col1:
                                final_minutes = st.number_input("Final step minutes", min_value=0, max_value=standard_duration, value=min(3, standard_duration))
                            with col2:
                                final_seconds = st.number_input("Final step seconds", min_value=0, max_value=59, value=0)
                                
                            final_duration = final_minutes + (final_seconds / 60)
                            
                            # Create step durations array
                            step_durations = [standard_duration] * (len(test_data) - 1) + [final_duration]
                        else:
                            # If all steps are complete, use standard duration for all
                            step_durations = [standard_duration] * len(test_data)
                        
                        # Add step durations to test data
                        test_data["StepDuration"] = step_durations
                
                st.dataframe(test_data)
                st.session_state.test_data = test_data
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error loading file: {e}")

def data_input_form(sport, resting_hr, resting_lactate):
    """Display manual data entry form with variable number of steps."""
    if sport == "Cycling":
        st.subheader("Cycling Test Data")
        
        # Allow user to specify the number of steps
        num_steps = st.number_input("Number of steps", min_value=1, max_value=20, value=16)
        
        # Create columns for power, HR, lactate and RPE
        cols = st.columns(4)
        with cols[0]:
            st.markdown("#### Power (Watts)")
            power_values = []
            for i in range(num_steps):  # User-defined number of steps
                if i == 0:
                    default_value = 0  # First step (rest)
                else:
                    # Calculate default stepped values (100W + 20W increments)
                    default_value = 100 + (i-1)*20
                power = st.number_input(f"Step {i+1}", key=f"power_{i}", min_value=0, value=default_value)
                if power > 0 or i == 0:  # Include rest step even if 0
                    power_values.append(power)
        
        with cols[1]:
            st.markdown("#### Heart Rate (bpm)")
            hr_values = []
            for i in range(len(power_values)):
                hr = st.number_input(f"Step {i+1}", key=f"hr_{i}", min_value=0, value=resting_hr + i*10)
                hr_values.append(hr)
        
        with cols[2]:
            st.markdown("#### Lactate (mmol/L)")
            lactate_values = []
            for i in range(len(power_values)):
                lactate = st.number_input(f"Step {i+1}", key=f"lactate_{i}", min_value=0.0, 
                                         value=resting_lactate + i*0.4 if i > 1 else resting_lactate, 
                                         step=0.1, format="%.1f")
                lactate_values.append(lactate)
        
        with cols[3]:
            st.markdown("#### RPE (6-20)")
            rpe_values = []
            for i in range(len(power_values)):
                rpe = st.number_input(f"Step {i+1}", key=f"rpe_{i}", min_value=6, max_value=20, value=min(6 + i*1, 20))
                rpe_values.append(rpe)
        
        # Add section for final step completion status
        if len(power_values) > 0:
            st.subheader("Final Step Completion")
            final_step_completed = st.checkbox("Was the final step completed fully?", value=False)
            
            if not final_step_completed:
                # If final step wasn't fully completed, allow entering the actual duration
                st.markdown("#### Duration of Final Step")
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    final_step_minutes = st.number_input("Minutes", min_value=0, max_value=60, value=2)
                with col2:
                    final_step_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=30)
                
                # Calculate the completion percentage
                standard_step_duration = 5  # Default 5 minutes per step
                final_step_duration = final_step_minutes + (final_step_seconds / 60)
                completion_percentage = (final_step_duration / standard_step_duration) * 100
                
                with col3:
                    st.markdown(f"**Completion: {completion_percentage:.1f}%** of standard step duration")
                
                # Calculate and show estimated effective power
                # For incomplete steps, power is typically higher than what was maintained
                effective_power = calculate_effective_intensity(power_values[-1], final_step_duration, standard_step_duration)
                adjustment_pct = ((effective_power / power_values[-1]) - 1) * 100
                
                st.info(f"""
                **Effective Power Calculation:**
                - Recorded Power: {power_values[-1]} watts for {final_step_duration:.1f} min (out of {standard_step_duration} min)
                - Estimated Effective Power: {effective_power:.1f} watts (+{adjustment_pct:.1f}%)
                
                *The effective power represents what you could likely have sustained for the full {standard_step_duration} min if the test had continued. This is calculated based on physiological models of fatigue accumulation and is used in threshold calculations.*
                """)
                
                # Option to use the recorded or effective power
                power_option = st.radio(
                    "Which power value would you like to use for analysis?",
                    ["Recorded Power", "Effective Power (Recommended for incomplete steps)"],
                    index=1
                )
                
                if power_option == "Effective Power (Recommended for incomplete steps)":
                    # Create a new column for this in the final dataframe
                    use_effective = True
                else:
                    use_effective = False
            else:
                # If final step was completed, set standard duration
                final_step_minutes = 5
                final_step_seconds = 0
                completion_percentage = 100
                
            # Add step durations for analysis
            step_durations = [5] * (len(power_values) - 1) + [final_step_minutes + (final_step_seconds / 60)]
                
        # Create dataframe
        if power_values:
            test_data = pd.DataFrame({
                "Power": power_values,
                "HeartRate": hr_values,
                "Lactate": lactate_values,
                "RPE": rpe_values
            })
            
            # Add step duration if available
            if 'step_durations' in locals():
                test_data["StepDuration"] = step_durations
                
                # Add effective power if final step wasn't completed
                if not final_step_completed and 'use_effective' in locals() and use_effective:
                    # We need to calculate effective power for the final step
                    test_data.loc[len(test_data)-1, "EffectivePower"] = effective_power
                    st.success(f"Using effective power of {effective_power:.1f} watts for final step in analysis.")
            
            # Show the test data
            st.subheader("Test Data")
            st.dataframe(test_data)
            st.session_state.test_data = test_data
    
    else:  # Running
        st.subheader("Running Test Data")
        
        # Allow user to specify the number of steps
        num_steps = st.number_input("Number of steps", min_value=1, max_value=20, value=16, key="run_num_steps")
        
        # Create columns for speed, HR, lactate and RPE
        cols = st.columns(4)
        with cols[0]:
            st.markdown("#### Speed (km/h)")
            speed_values = []
            for i in range(num_steps):
                if i == 0:
                    default_value = 0.0  # First step (rest)
                else:
                    default_value = 8.0 + (i-1)*0.5  # Calculate stepped values
                speed = st.number_input(f"Step {i+1}", key=f"speed_{i}", min_value=0.0, value=default_value, step=0.1, format="%.1f")
                if speed > 0 or i == 0:  # Include rest step even if 0
                    speed_values.append(speed)
        
        with cols[1]:
            st.markdown("#### Heart Rate (bpm)")
            hr_values = []
            for i in range(len(speed_values)):
                hr = st.number_input(f"Step {i+1}", key=f"run_hr_{i}", min_value=0, value=resting_hr + i*10)
                hr_values.append(hr)
        
        with cols[2]:
            st.markdown("#### Lactate (mmol/L)")
            lactate_values = []
            for i in range(len(speed_values)):
                lactate = st.number_input(f"Step {i+1}", key=f"run_lactate_{i}", min_value=0.0, 
                                         value=resting_lactate + i*0.3 if i > 1 else resting_lactate, 
                                         step=0.1, format="%.1f")
                lactate_values.append(lactate)
        
        with cols[3]:
            st.markdown("#### RPE (6-20)")
            rpe_values = []
            for i in range(len(speed_values)):
                rpe = st.number_input(f"Step {i+1}", key=f"run_rpe_{i}", min_value=6, max_value=20, value=min(6 + i*1, 20))
                rpe_values.append(rpe)
        
        # Add section for final step completion status
        if len(speed_values) > 0:
            st.subheader("Final Step Completion")
            final_step_completed = st.checkbox("Was the final step completed fully?", value=False, key="running_final_step")
            
            if not final_step_completed:
                # If final step wasn't fully completed, allow entering the actual duration
                st.markdown("#### Duration of Final Step")
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    final_step_minutes = st.number_input("Minutes", min_value=0, max_value=60, value=2, key="run_minutes")
                with col2:
                    final_step_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=30, key="run_seconds")
                
                # Calculate the completion percentage
                standard_step_duration = 4  # Default 4 minutes per step for running
                final_step_duration = final_step_minutes + (final_step_seconds / 60)
                completion_percentage = (final_step_duration / standard_step_duration) * 100
                
                with col3:
                    st.markdown(f"**Completion: {completion_percentage:.1f}%** of standard step duration")
                
                # Calculate and show estimated effective speed
                effective_speed = calculate_effective_intensity(speed_values[-1], final_step_duration, standard_step_duration)
                adjustment_pct = ((effective_speed / speed_values[-1]) - 1) * 100
                
                # Convert to pace for display
                effective_pace = 60 / effective_speed
                effective_pace_min = int(effective_pace)
                effective_pace_sec = int((effective_pace - effective_pace_min) * 60)
                
                # Current pace
                current_pace = 60 / speed_values[-1]
                current_pace_min = int(current_pace)
                current_pace_sec = int((current_pace - current_pace_min) * 60)
                
                st.info(f"""
                **Effective Speed Calculation:**
                - Recorded Speed: {speed_values[-1]:.2f} km/h ({current_pace_min}:{current_pace_sec:02d} min/km) for {final_step_duration:.1f} min (out of {standard_step_duration} min)
                - Estimated Effective Speed: {effective_speed:.2f} km/h ({effective_pace_min}:{effective_pace_sec:02d} min/km) (+{adjustment_pct:.1f}%)
                
                *The effective speed represents what you could likely have sustained for the full {standard_step_duration} min if the test had continued. This is calculated based on physiological models of fatigue accumulation and is used in threshold calculations.*
                """)
                
                # Option to use the recorded or effective speed
                speed_option = st.radio(
                    "Which speed value would you like to use for analysis?",
                    ["Recorded Speed", "Effective Speed (Recommended for incomplete steps)"],
                    index=1,
                    key="speed_option"
                )
                
                if speed_option == "Effective Speed (Recommended for incomplete steps)":
                    # Create a new column for this in the final dataframe
                    use_effective = True
                else:
                    use_effective = False
            else:
                # If final step was completed, set standard duration
                final_step_minutes = 4
                final_step_seconds = 0
                completion_percentage = 100
                
            # Add step durations for analysis
            step_durations = [4] * (len(speed_values) - 1) + [final_step_minutes + (final_step_seconds / 60)]
                
        # Create dataframe
        if speed_values:
            test_data = pd.DataFrame({
                "Speed": speed_values,
                "HeartRate": hr_values,
                "Lactate": lactate_values,
                "RPE": rpe_values
            })
            
            # Add step duration if available
            if 'step_durations' in locals():
                test_data["StepDuration"] = step_durations
                
                # Add effective speed if final step wasn't completed
                if not final_step_completed and 'use_effective' in locals() and use_effective:
                    # We need to calculate effective speed for the final step
                    test_data.loc[len(test_data)-1, "EffectiveSpeed"] = effective_speed
                    
                    # Convert to pace for display
                    effective_pace = 60 / effective_speed
                    effective_pace_min = int(effective_pace)
                    effective_pace_sec = int((effective_pace - effective_pace_min) * 60)
                    
                    st.success(f"Using effective speed of {effective_speed:.2f} km/h ({effective_pace_min}:{effective_pace_sec:02d} min/km) for final step in analysis.")
            
            # Add pace (min/km) calculation
            if not test_data.empty:
                test_data["Pace"] = test_data["Speed"].apply(lambda x: f"{int(60/x)}:{int((60/x - int(60/x))*60):02d}" if x > 0 else "0:00")
                
                # Add effective pace if exists
                if "EffectiveSpeed" in test_data.columns:
                    test_data["EffectivePace"] = test_data["EffectiveSpeed"].apply(
                        lambda x: f"{int(60/x)}:{int((60/x - int(60/x))*60):02d}" if x > 0 else "0:00"
                    )
            
            # Show the test data
            st.subheader("Test Data")
            st.dataframe(test_data)
            st.session_state.test_data = test_data

# Now let's fix the display_results function to address the DataFrame styling error
def display_results(sport, athlete_info):
    """Display threshold analysis results."""
    st.header("Threshold Analysis")
    
    # Choose threshold calculation methods
    st.subheader("Calculation Methods")
    calculation_methods = st.multiselect(
        "Select Methods", 
        [
            "Modified Dmax", 
            "Lactate Turnpoint", 
            "4 mmol/L Fixed Threshold", 
            "Individual Anaerobic Threshold",
            "Critical Power" if sport == "Cycling" else None
        ],
        default=["Modified Dmax"]
    )
    
    # Remove None values (Critical Power for running)
    calculation_methods = [method for method in calculation_methods if method]
    
    if st.button("Calculate Thresholds"):
        # Process data and calculate thresholds
        processed_data = process_input_data(st.session_state.test_data, sport)
        
        if sport == "Cycling":
            # Check if we have effective power values from incomplete steps
            if "EffectivePower" in processed_data.columns:
                x_column = "EffectivePower"
                x_label = "Effective Power (Watts)"
                st.info("Using effective power values adjusted for incomplete steps.")
            else:
                x_column = "Power"
                x_label = "Power (Watts)"
        else:
            # Check if we have effective speed values from incomplete steps
            if "EffectiveSpeed" in processed_data.columns:
                x_column = "EffectiveSpeed"
                x_label = "Effective Speed (km/h)"
                st.info("Using effective speed values adjusted for incomplete steps.")
            else:
                x_column = "Speed"
                x_label = "Speed (km/h)"
        
        results = {}
        
        if "Modified Dmax" in calculation_methods:
            threshold, details = calculate_modified_dmax(
                processed_data[x_column].values, 
                processed_data["Lactate"].values,
                athlete_info.get('resting_lactate', 0.8)
            )
            results["Modified Dmax"] = {
                "threshold": threshold,
                "details": details
            }
        
        if "Lactate Turnpoint" in calculation_methods:
            threshold, details = calculate_lactate_turnpoint(
                processed_data[x_column].values, 
                processed_data["Lactate"].values
            )
            results["Lactate Turnpoint"] = {
                "threshold": threshold,
                "details": details
            }
        
        if "4 mmol/L Fixed Threshold" in calculation_methods:
            threshold, details = calculate_fixed_threshold(
                processed_data[x_column].values, 
                processed_data["Lactate"].values,
                threshold_value=4.0
            )
            results["4 mmol/L Fixed Threshold"] = {
                "threshold": threshold,
                "details": details
            }
        
        if "Individual Anaerobic Threshold" in calculation_methods:
            threshold, details = calculate_individual_anaerobic_threshold(
                processed_data[x_column].values, 
                processed_data["Lactate"].values,
                athlete_info.get('resting_lactate', 0.8)
            )
            results["Individual Anaerobic Threshold"] = {
                "threshold": threshold,
                "details": details
            }
        
        if "Critical Power" in calculation_methods and sport == "Cycling":
            threshold, details = calculate_critical_power(
                processed_data["Power"].values, 
                processed_data["Lactate"].values
            )
            results["Critical Power"] = {
                "threshold": threshold,
                "details": details
            }
        
        # Display results
        st.subheader("Results")
        
        # Display lactate curve plot
        fig = create_lactate_curve_plot(
            processed_data[x_column].values,
            processed_data["Lactate"].values,
            processed_data["HeartRate"].values,
            results,
            x_label=x_label,
            sport=sport
        )
        st.pyplot(fig)
        
        # Display interactive plot
        interactive_chart = create_interactive_plot(
            processed_data,
            results,
            x_column=x_column,
            sport=sport
        )
        st.plotly_chart(interactive_chart)
        
        # Display threshold values
        st.subheader("Threshold Values")
        
        # Set up columns to display results side by side
        cols = st.columns(len(results))
        
        for i, (method_name, result) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"#### {method_name}")
                
                # Check if this uses effective intensity
                uses_effective = result['details'].get('uses_effective_intensity', False)
                effective_label = " (Effective)" if uses_effective else ""
                
                # Format threshold value appropriately
                if sport == "Cycling":
                    threshold_str = f"{result['threshold']:.1f} W{effective_label}"
                    threshold_rel = f"{result['threshold']/athlete_info.get('weight', 70):.2f} W/kg"
                    st.markdown(f"**Threshold:** {threshold_str}")
                    st.markdown(f"**Relative:** {threshold_rel}")
                else:
                    threshold_str = f"{result['threshold']:.2f} km/h{effective_label}"
                    pace_min = int(60 / result['threshold'])
                    pace_sec = int((60 / result['threshold'] - pace_min) * 60)
                    pace_str = f"{pace_min}:{pace_sec:02d} min/km"
                    st.markdown(f"**Threshold:** {threshold_str}")
                    st.markdown(f"**Pace:** {pace_str}")
                
                # Display heart rate at threshold if available
                if 'hr_at_threshold' in result['details'] and result['details']['hr_at_threshold'] is not None:
                    st.markdown(f"**HR:** {result['details']['hr_at_threshold']:.0f} bpm")
        
        # Calculate and display training zones
        st.subheader("Training Zones")
        
        # Choose a method for zone calculation
        zone_method = st.selectbox(
            "Select threshold method for zone calculation",
            list(results.keys())
        )
        
        threshold_value = results[zone_method]["threshold"]
        
        training_zones = calculate_training_zones(
            threshold_value, 
            sport, 
            athlete_info.get('max_hr'), 
            results[zone_method]["details"].get("hr_at_threshold")
        )
        
        # Display zones in a table with proper styling approach that's compatible with newer pandas
        zones_df = pd.DataFrame(training_zones)
        
        # Fix for the styling error - use a method that's compatible with newer pandas versions
        try:
            # Try to use set_properties instead of apply
            st.dataframe(zones_df.style.set_properties(**{'background-color': '#f0f0f0'}))
        except Exception as e:
            # If that fails too, just display without styling
            st.dataframe(zones_df)
            st.warning("Note: Zone styling could not be applied due to pandas version compatibility.")
        
        # Store results for report generation
        st.session_state.results = results
        st.session_state.training_zones = training_zones
        st.session_state.processed_data = processed_data
        st.session_state.sport = sport

def export_report(athlete_info):
    """Display report export options and generate PDF."""
    st.header("Generate Report")
    
    st.subheader("Report Settings")
    
    include_logo = st.checkbox("Include Company Logo", value=True)
    include_training_zones = st.checkbox("Include Training Zones", value=True)
    include_raw_data = st.checkbox("Include Raw Test Data", value=False)
    
    additional_notes = st.text_area("Additional Notes", height=100)
    
    if st.button("Generate PDF Report"):
        try:
            with st.spinner("Generating PDF report..."):
                report_pdf = generate_pdf_report(
                    athlete_info,
                    st.session_state.processed_data,
                    st.session_state.results,
                    st.session_state.training_zones,
                    st.session_state.sport,
                    additional_notes=additional_notes,
                    include_logo=include_logo,
                    include_training_zones=include_training_zones,
                    include_raw_data=include_raw_data
                )
            
            # Create download link for PDF
            b64 = base64.b64encode(report_pdf).decode("utf-8")
            report_filename = f"{athlete_info.get('name', 'Athlete').replace(' ', '_')}_{st.session_state.sport}_Threshold_Report.pdf"
            href = f'<a href="data:application/pdf;base64,{b64}" download="{report_filename}">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Preview note
            st.success("PDF generated successfully! Click the link above to download.")
            st.info("Note: PDF preview may not display correctly in Streamlit. Please download the file to view it.")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            st.info("Try disabling some options like 'Include Logo' and try again.")

def main():
    """Main application function."""
    # App Header
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            # Try different potential logo paths
            logo_paths = [
                "assets/logo.png",
                "assets/Logotype_Light@2x.png",
                "/workspaces/threshold_analyzer/assets/logo.png",
                "/workspaces/threshold_analyzer/assets/Logotype_Light@2x.png"
            ]
            
            logo_loaded = False
            for logo_path in logo_paths:
                try:
                    st.image(logo_path, width=150)
                    logo_loaded = True
                    break
                except:
                    continue
            
            if not logo_loaded:
                # If no logo is found, display a placeholder
                st.markdown("### Lindblom Coaching")
        except:
            st.markdown("### Lindblom Coaching")
    with col2:
        st.title("Threshold Analyzer")
        st.markdown("#### Professional threshold analysis for cycling and running")

    # Sidebar for athlete information
    with st.sidebar:
        st.header("Athlete Information")
        
        athlete_name = st.text_input("Athlete Name")
        dob = st.date_input("Date of Birth")
        age = (datetime.now().date() - dob).days // 365 if dob else None
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=175)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        
        sport = st.selectbox("Sport", ["Cycling", "Running"])
        test_date = st.date_input("Test Date", datetime.now().date())
        
        resting_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=100, value=60)
        max_hr = st.number_input("Maximum Heart Rate", min_value=120, max_value=220, value=185)
        resting_lactate = st.number_input("Resting Lactate (mmol/L)", min_value=0.0, max_value=3.0, value=0.8, step=0.1)
        
        athlete_info = {
            "name": athlete_name,
            "dob": dob,
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "resting_hr": resting_hr,
            "max_hr": max_hr,
            "resting_lactate": resting_lactate,
            "test_date": test_date
        }
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Data Input", "Analysis & Results", "Export Report"])
    
    # Tab 1: Data Input
    with tab1:
        st.header("Test Data Input")
        
        input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV/Excel"])
        
        if input_method == "Manual Entry":
            data_input_form(sport, resting_hr, resting_lactate)
        else:
            data_upload_form(sport)
    
    # Tab 2: Analysis & Results
    with tab2:
        if 'test_data' in st.session_state and st.session_state.test_data is not None:
            display_results(sport, athlete_info)
        else:
            st.info("Please enter test data in the 'Data Input' tab first")
    
    # Tab 3: Export Report
    with tab3:
        if 'results' in st.session_state:
            export_report(athlete_info)
        else:
            st.info("Complete the analysis in the 'Analysis & Results' tab to generate a report")
    
    # Footer
    st.markdown("""
    <footer>
        <p>© Lindblom Coaching - Professional Threshold Analysis</p>
    </footer>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
