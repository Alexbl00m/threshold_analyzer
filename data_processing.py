import pandas as pd
import numpy as np


def validate_data(data, sport):
    """
    Validates that the input data has the required columns and structure.
    
    Args:
        data: DataFrame containing test data
        sport: Either "Cycling" or "Running"
        
    Returns:
        is_valid: Boolean indicating if the data is valid
        message: Error message if data is invalid, otherwise empty string
    """
    # Check if DataFrame is empty
    if data.empty:
        return False, "Data is empty"
    
    # Check required columns based on sport
    if sport == "Cycling":
        required_columns = ["Power", "Lactate"]
    else:  # Running
        required_columns = ["Speed", "Lactate"]
    
    # Optional columns
    optional_columns = ["HeartRate", "RPE", "StepDuration", "Pace"]
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if there are enough data points (at least 3)
    if len(data) < 3:
        return False, "At least 3 data points are required for analysis"
    
    # Check if power/speed values are positive
    intensity_col = "Power" if sport == "Cycling" else "Speed"
    if (data[intensity_col] <= 0).any():
        return False, f"{intensity_col} values must be positive"
    
    # Check if lactate values are non-negative
    if (data["Lactate"] < 0).any():
        return False, "Lactate values cannot be negative"
    
    # Check step duration values if present
    if "StepDuration" in data.columns:
        if (data["StepDuration"] <= 0).any():
            return False, "Step duration values must be positive"
    
    # All checks passed
    return True, ""


def calculate_effective_intensity(intensity, step_duration, standard_duration=5):
    """
    Calculates the effective intensity (power/speed) for incomplete steps.
    
    For incomplete steps, the effective intensity is typically higher than the
    average due to fatigue. This implements a simple model where:
    - At 100% completion, effective intensity = actual intensity
    - At lower completion percentages, there's a small boost to reflect
      what the athlete could have sustained for the full duration
    
    Args:
        intensity: The recorded intensity value (power in watts or speed in km/h)
        step_duration: Actual duration of the step in minutes
        standard_duration: Standard step duration in minutes (default: 5)
        
    Returns:
        effective_intensity: Adjusted intensity value
    """
    if step_duration >= standard_duration:
        # If step was completed fully, no adjustment needed
        return intensity
    
    # Calculate completion percentage
    completion_percentage = step_duration / standard_duration
    
    # Apply an adjustment factor that increases as completion percentage decreases
    # The adjustment is higher for very short durations and minimal for near-complete steps
    adjustment_factor = 1 + max(0, (1 - completion_percentage)) * 0.15
    
    # Calculate effective intensity
    effective_intensity = intensity * adjustment_factor
    
    return effective_intensity


def process_input_data(data, sport):
    """
    Processes and prepares raw input data for analysis.
    
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
        
        # Add a note about the adjustment to incomplete steps
        incomplete_steps = processed_data[processed_data['StepDuration'] < standard_duration]
        if not incomplete_steps.empty:
            intensity_unit = "W" if sport == "Cycling" else "km/h"
            for _, row in incomplete_steps.iterrows():
                print(f"Step at {row[intensity_col]} {intensity_unit} was adjusted to effective {row[f'Effective{intensity_col}']:.1f} {intensity_unit} based on {row['StepDuration']:.2f} min duration")
    
    # Ensure there are no duplicate intensity values (average if there are)
    if processed_data[intensity_col].duplicated().any():
        # Group by intensity and average other columns
        agg_dict = {
            'Lactate': 'mean',
            'RPE': 'mean' if 'RPE' in processed_data.columns else None,
            'StepDuration': 'mean'
        }
        
        # Include HeartRate if available
        if 'HeartRate' in processed_data.columns:
            agg_dict['HeartRate'] = 'mean'
        
        # Include EffectiveIntensity if available
        effective_col = f'Effective{intensity_col}'
        if effective_col in processed_data.columns:
            agg_dict[effective_col] = 'mean'
            
        # Remove None values
        agg_dict = {k: v for k, v in agg_dict.items() if v is not None}
        
        # Perform groupby aggregation
        processed_data = processed_data.groupby(intensity_col).agg(agg_dict).reset_index()
    
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


def pace_to_speed(pace):
    """
    Converts pace (min:sec per km) to speed (km/h).
    
    Args:
        pace: String in format "MM:SS"
        
    Returns:
        speed: Speed in km/h
    """
    try:
        minutes, seconds = map(int, pace.split(':'))
        total_seconds = minutes * 60 + seconds
        speed = 3600 / total_seconds  # km/h
        return speed
    except (ValueError, ZeroDivisionError):
        return 0
    

def speed_to_pace(speed):
    """
    Converts speed (km/h) to pace (min:sec per km).
    
    Args:
        speed: Speed in km/h
        
    Returns:
        pace: String in format "MM:SS"
    """
    try:
        if speed <= 0:
            return "0:00"
        
        # 60 min / speed (km/h) = min/km
        minutes_per_km = 60 / speed
        minutes = int(minutes_per_km)
        seconds = int((minutes_per_km - minutes) * 60)
        
        return f"{minutes}:{seconds:02d}"
    except (ValueError, ZeroDivisionError):
        return "0:00"


def estimate_vo2max(threshold_value, weight, sport, gender="Male"):
    """
    Estimates VO2max based on threshold value (power or speed).
    This is a simplified model and actual VO2max would require lab testing.
    
    Args:
        threshold_value: Threshold power (W) or speed (km/h)
        weight: Athlete weight in kg
        sport: Either "Cycling" or "Running"
        gender: Either "Male" or "Female"
        
    Returns:
        vo2max: Estimated VO2max in ml/kg/min
    """
    if sport == "Cycling":
        # Cycling-specific formula based on FTP
        # FTP to VO2max estimation (approximate)
        gender_factor = 1.0 if gender == "Male" else 0.9
        vo2max = (10.8 * threshold_value / weight + 7) * gender_factor
    else:
        # Running-specific formula based on threshold speed
        # vVO2max estimation based on threshold speed (approximate)
        gender_factor = 1.0 if gender == "Male" else 0.92
        speed_in_ms = threshold_value / 3.6  # Convert km/h to m/s
        vo2max = (4.5 * speed_in_ms + 3.5) * gender_factor
    
    return vo2max


def calculate_ftp_from_threshold(threshold_value, threshold_method):
    """
    Estimates FTP (Functional Threshold Power) from different threshold methods.
    
    Args:
        threshold_value: Threshold power in watts
        threshold_method: Method used to determine threshold
        
    Returns:
        ftp: Estimated FTP in watts
    """
    if threshold_method == "4 mmol/L Fixed Threshold":
        # FTP is typically 95% of power at 4 mmol/L
        ftp = threshold_value * 0.95
    elif threshold_method == "Modified Dmax":
        # FTP is approximately equal to Modified Dmax in many cases
        ftp = threshold_value * 0.97
    elif threshold_method == "Lactate Turnpoint":
        # Lactate turnpoint is often slightly above FTP
        ftp = threshold_value * 0.98
    elif threshold_method == "Individual Anaerobic Threshold":
        # IAT is often very close to FTP
        ftp = threshold_value * 0.99
    elif threshold_method == "Critical Power":
        # CP is typically very slightly above FTP
        ftp = threshold_value * 0.96
    else:
        # Default case
        ftp = threshold_value * 0.95
    
    return ftp
