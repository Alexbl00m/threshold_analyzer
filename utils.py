import numpy as np


def calculate_training_zones(threshold_value, sport, max_hr=None, threshold_hr=None):
    """
    Calculates training zones based on threshold value.
    
    Args:
        threshold_value: Threshold power (W) or speed (km/h)
        sport: Either "Cycling" or "Running"
        max_hr: Maximum heart rate (if available)
        threshold_hr: Heart rate at threshold (if available)
        
    Returns:
        zones: List of dictionaries with zone information
    """
    zones = []
    
    if sport == "Cycling":
        # Cycling zones based on percentage of threshold power (FTP)
        # Using common zone definitions from TrainingPeaks/British Cycling
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
        
        # Add heart rate zones if available
        if threshold_hr is not None:
            # Use heart rate at threshold for calculating zones
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
        # Convert threshold speed to pace (min:sec per km)
        threshold_pace_mins = int(60 / threshold_value)
        threshold_pace_secs = int(60 * (60 / threshold_value - threshold_pace_mins))
        threshold_pace = f"{threshold_pace_mins}:{threshold_pace_secs:02d}"
        
        # Running zones based on percentage of threshold pace
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
        
        # Add heart rate zones if available
        if threshold_hr is not None:
            # Use heart rate at threshold for calculating zones
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
    """
    Converts speed range to pace range (min:sec per km).
    
    Args:
        speed_low: Lower bound of speed range (km/h)
        speed_high: Upper bound of speed range (km/h), can be None for open-ended ranges
        slower: If True, format as "slower than X:XX"
        
    Returns:
        pace_range: String representation of pace range
    """
    if speed_low <= 0:
        return "N/A"
    
    # Convert lower bound to pace
    mins_low = int(60 / speed_low)
    secs_low = int(60 * (60 / speed_low - mins_low))
    pace_low = f"{mins_low}:{secs_low:02d}"
    
    # If we have a high bound, calculate that pace too
    if speed_high is not None:
        mins_high = int(60 / speed_high)
        secs_high = int(60 * (60 / speed_high - mins_high))
        pace_high = f"{mins_high}:{secs_high:02d}"
        
        # Note: faster pace = lower numbers, so we reverse the order
        if slower:
            return f"Slower than {pace_high}"
        else:
            return f"{pace_high} - {pace_low}"
    else:
        if slower:
            return f"Slower than {pace_low}"
        else:
            return f"Faster than {pace_low}"


def estimate_lactate_for_intensity(intensity, lactate_data, intensity_column, lactate_column):
    """
    Estimates the lactate value for a given intensity using interpolation.
    
    Args:
        intensity: The intensity value to estimate lactate for
        lactate_data: DataFrame with lactate test data
        intensity_column: Column name for intensity values
        lactate_column: Column name for lactate values
        
    Returns:
        estimated_lactate: Estimated lactate value
    """
    from scipy import interpolate
    
    # Sort by intensity
    sorted_data = lactate_data.sort_values(by=intensity_column)
    
    # Create interpolation function
    f = interpolate.interp1d(
        sorted_data[intensity_column],
        sorted_data[lactate_column],
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    
    # Estimate lactate value
    estimated_lactate = float(f(intensity))
    
    return max(0, estimated_lactate)  # Ensure non-negative value


def estimate_cycling_speed(power, weight, cda=0.36, crr=0.004, elevation_gain=0, distance=40):
    """
    Estimates cycling speed based on power, weight, and environmental factors.
    This is a simplified model based on basic physics.
    
    Args:
        power: Power in watts
        weight: Rider + bike weight in kg
        cda: Coefficient of drag area (default: 0.36 for typical road bike position)
        crr: Coefficient of rolling resistance (default: 0.004 for good road tires)
        elevation_gain: Elevation gain in meters (default: 0 for flat course)
        distance: Distance in km (default: 40)
        
    Returns:
        speed: Estimated speed in km/h
    """
    # Constants
    rho = 1.225  # Air density at sea level (kg/m^3)
    g = 9.81     # Gravitational acceleration (m/s^2)
    
    # Elevation gain as percentage
    grade = elevation_gain / (distance * 1000) * 100 if distance > 0 else 0
    
    # Simple physics-based model
    # P = F * v, where F is the sum of forces (drag, rolling resistance, gravity)
    # This is a simplified approach just for estimation purposes
    
    if power <= 0:
        return 0
    
    # For a flat course with no wind, a simplified model:
    # P = (0.5 * rho * CdA * v^3) + (Crr * m * g * v)
    # We use a cubic approximation to solve for v
    
    # For simplicity, let's use an approximation formula based on typical values
    # This won't be exact but gives a reasonable estimate for typical road cycling
    
    # Coefficient for converting W/kg to km/h (empirical)
    if grade <= 0:  # Flat or downhill
        speed = (power / (weight + 8)) * 2.9  # Simplified approximation
    else:  # Uphill
        # Uphill speed is more determined by W/kg
        w_per_kg = power / weight
        speed = w_per_kg * (12 - grade * 0.5)  # Rough approximation that decreases with grade
    
    return max(5, min(60, speed))  # Cap between 5 and 60 km/h for realism


def format_time(minutes):
    """
    Formats time in minutes to hours:minutes:seconds.
    
    Args:
        minutes: Time in minutes
        
    Returns:
        formatted_time: Time in format HH:MM:SS or MM:SS
    """
    hours = int(minutes / 60)
    mins = int(minutes) % 60
    secs = int((minutes * 60) % 60)
    
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    else:
        return f"{mins}:{secs:02d}"


def format_pace(minutes_per_km):
    """
    Formats pace in minutes per km to minutes:seconds.
    
    Args:
        minutes_per_km: Pace in minutes per km
        
    Returns:
        formatted_pace: Pace in format MM:SS
    """
    mins = int(minutes_per_km)
    secs = int((minutes_per_km - mins) * 60)
    
    return f"{mins}:{secs:02d}"


def calculate_bmi(weight, height):
    """
    Calculates Body Mass Index (BMI).
    
    Args:
        weight: Weight in kg
        height: Height in cm
        
    Returns:
        bmi: Body Mass Index
    """
    if height <= 0 or weight <= 0:
        return 0
    
    # BMI = weight(kg) / height(m)^2
    height_m = height / 100  # Convert cm to m
    bmi = weight / (height_m * height_m)
    
    return bmi


def get_bmi_category(bmi):
    """
    Gets BMI category based on BMI value.
    
    Args:
        bmi: Body Mass Index
        
    Returns:
        category: BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"
