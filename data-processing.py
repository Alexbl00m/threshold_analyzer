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
    optional_columns = ["HeartRate", "RPE"]
    
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
    
    # All checks passed
    return True, ""


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
    
    # Sort by intensity
    intensity_col = "Power" if sport == "Cycling" else "Speed"
    processed_data = processed_data.sort_values(by=intensity_col)
    
    # Ensure there are no duplicate intensity values (average if there are)
    if processed_data[intensity_col].duplicated().any():
        # Group by intensity and average other columns
        processed_data = processed_data.groupby(intensity_col).agg({
            'Lactate': 'mean',
            'HeartRate': 'mean' if 'HeartRate' in processed_data.columns else None,
            'RPE': 'mean' if 'RPE' in processed_data.columns else None
        }).reset_index()
    
    # Handle missing HR data if needed
    if 'HeartRate' not in processed_data.columns:
        processed_data['HeartRate'] = None
    
    # Add pace calculation for running
    if sport == "Running" and 'Pace' not in processed_data.columns:
        processed_data['Pace'] = processed_data['Speed'].apply(
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
