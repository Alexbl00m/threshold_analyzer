import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
import sys

# Add the current directory to Python's path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now try the imports
try:
    from threshold_models import (
        calculate_modified_dmax, 
        calculate_lactate_turnpoint, 
        calculate_fixed_threshold,
        calculate_individual_anaerobic_threshold,
        calculate_critical_power
    )
    from data_processing import process_input_data, validate_data
    from visualization import create_lactate_curve_plot, create_interactive_plot
    from pdf_generator import generate_pdf_report
    from utils import calculate_training_zones
    
    # No need for debug info if imports succeed
    imports_succeeded = True
except ImportError as e:
    st.error(f"Import error: {e}")
    # Additional debug info
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Files in current directory: {os.listdir('.')}")
    st.write(f"Python path: {sys.path}")
    
    imports_succeeded = False

# Proceed with the application only if imports succeeded
if imports_succeeded:

# Set page configuration
st.set_page_config(
    page_title="Threshold Analyzer - Lindblom Coaching",
    page_icon="🏊‍♂️🚴‍♂️🏃‍♂️",
    layout="wide"
)

# Custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Sidebar for inputs
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

# Main area - Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Data Input", "Analysis & Results", "Export Report"])

# Tab 1: Data Input
with tab1:
    st.header("Test Data Input")
    
    input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV/Excel"])
    
    if input_method == "Manual Entry":
        if sport == "Cycling":
            st.subheader("Cycling Test Data")
            
            # Create columns for power, HR, lactate and RPE
            cols = st.columns(4)
            with cols[0]:
                st.markdown("#### Power (Watts)")
                power_values = []
                for i in range(10):
                    power = st.number_input(f"Step {i+1}", key=f"power_{i}", min_value=0, value=0 if i == 0 else 100 + i*20)
                    if power > 0:
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
                    lactate = st.number_input(f"Step {i+1}", key=f"lactate_{i}", min_value=0.0, value=resting_lactate + i*0.4 if i > 1 else resting_lactate, step=0.1)
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
                final_step_completed = st.checkbox("Was the final step completed fully?", value=True)
                
                if not final_step_completed:
                    # If final step wasn't fully completed, allow entering the actual duration
                    st.markdown("#### Duration of Final Step")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        final_step_minutes = st.number_input("Minutes", min_value=0, max_value=60, value=3)
                    with col2:
                        final_step_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0)
                    
                    # Calculate the completion percentage
                    standard_step_duration = 5  # Default 5 minutes per step
                    final_step_duration = final_step_minutes + (final_step_seconds / 60)
                    completion_percentage = (final_step_duration / standard_step_duration) * 100
                    
                    with col3:
                        st.markdown(f"**Completion: {completion_percentage:.1f}%** of standard step duration")
                    
                    # Adjust the final power if needed
                    if st.checkbox("Adjust final power based on completion percentage"):
                        # For incomplete steps, power is typically slightly higher than what was maintained
                        adjusted_power = int(power_values[-1] * (1 + (1 - completion_percentage/100) * 0.1))
                        st.markdown(f"**Adjusted final power: {adjusted_power} watts** (estimation based on {completion_percentage:.1f}% completion)")
                        
                        # Option to use the adjusted power
                        if st.button("Use adjusted power"):
                            power_values[-1] = adjusted_power
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
                
                st.dataframe(test_data)
                
        else:  # Running
            st.subheader("Running Test Data")
            
            # Create columns for speed, HR, lactate and RPE
            cols = st.columns(4)
            with cols[0]:
                st.markdown("#### Speed (km/h)")
                speed_values = []
                for i in range(10):
                    speed = st.number_input(f"Step {i+1}", key=f"speed_{i}", min_value=0.0, value=0.0 if i == 0 else 8.0 + i*0.5, step=0.1)
                    if speed > 0:
                        speed_values.append(speed)
            
            with cols[1]:
                st.markdown("#### Heart Rate (bpm)")
                hr_values = []
                for i in range(len(speed_values)):
                    hr = st.number_input(f"Step {i+1}", key=f"hr_{i}", min_value=0, value=resting_hr + i*10)
                    hr_values.append(hr)
            
            with cols[2]:
                st.markdown("#### Lactate (mmol/L)")
                lactate_values = []
                for i in range(len(speed_values)):
                    lactate = st.number_input(f"Step {i+1}", key=f"lactate_{i}", min_value=0.0, value=resting_lactate + i*0.3 if i > 1 else resting_lactate, step=0.1)
                    lactate_values.append(lactate)
            
            with cols[3]:
                st.markdown("#### RPE (6-20)")
                rpe_values = []
                for i in range(len(speed_values)):
                    rpe = st.number_input(f"Step {i+1}", key=f"rpe_{i}", min_value=6, max_value=20, value=min(6 + i*1, 20))
                    rpe_values.append(rpe)
            
            # Add section for final step completion status
            if len(speed_values) > 0:
                st.subheader("Final Step Completion")
                final_step_completed = st.checkbox("Was the final step completed fully?", value=True, key="running_final_step")
                
                if not final_step_completed:
                    # If final step wasn't fully completed, allow entering the actual duration
                    st.markdown("#### Duration of Final Step")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    with col1:
                        final_step_minutes = st.number_input("Minutes", min_value=0, max_value=60, value=3, key="run_minutes")
                    with col2:
                        final_step_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, key="run_seconds")
                    
                    # Calculate the completion percentage
                    standard_step_duration = 4  # Default 4 minutes per step for running
                    final_step_duration = final_step_minutes + (final_step_seconds / 60)
                    completion_percentage = (final_step_duration / standard_step_duration) * 100
                    
                    with col3:
                        st.markdown(f"**Completion: {completion_percentage:.1f}%** of standard step duration")
                    
                    # Adjust the final speed if needed
                    if st.checkbox("Adjust final speed based on completion percentage", key="adjust_speed"):
                        # For incomplete steps, speed is typically slightly higher than what was maintained
                        adjusted_speed = speed_values[-1] * (1 + (1 - completion_percentage/100) * 0.05)
                        adjusted_pace = 60 / adjusted_speed
                        adjusted_pace_min = int(adjusted_pace)
                        adjusted_pace_sec = int((adjusted_pace - adjusted_pace_min) * 60)
                        
                        st.markdown(f"**Adjusted final speed: {adjusted_speed:.2f} km/h** ({adjusted_pace_min}:{adjusted_pace_sec:02d} min/km) \
                                    (estimation based on {completion_percentage:.1f}% completion)")
                        
                        # Option to use the adjusted speed
                        if st.button("Use adjusted speed"):
                            speed_values[-1] = adjusted_speed
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
                
                # Add pace (min/km) calculation
                if not test_data.empty:
                    test_data["Pace"] = test_data["Speed"].apply(lambda x: f"{int(60/x)}:{int((60/x - int(60/x))*60):02d}" if x > 0 else "0:00")
                
                st.dataframe(test_data)

    else:  # Upload CSV/Excel
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
                else:
                    st.error(message)
                    test_data = None
            except Exception as e:
                st.error(f"Error loading file: {e}")
                test_data = None

# Tab 2: Analysis & Results
with tab2:
    if 'test_data' in locals() and test_data is not None and not test_data.empty:
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
                "Critical Power"
            ],
            default=["Modified Dmax"]
        )
        
        if st.button("Calculate Thresholds"):
            # Process data and calculate thresholds
            processed_data = process_input_data(test_data, sport)
            
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
                    resting_lactate
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
                    resting_lactate
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
                        threshold_rel = f"{result['threshold']/weight:.2f} W/kg"
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
                max_hr, 
                results[zone_method]["details"].get("hr_at_threshold")
            )
            
            # Display zones in a table with colored formatting
            zones_df = pd.DataFrame(training_zones)
            st.dataframe(zones_df.style.applymap(lambda _: 'background-color: #f0f0f0'))
            
            # Store results for report generation
            st.session_state.results = results
            st.session_state.training_zones = training_zones
            st.session_state.processed_data = processed_data
            st.session_state.sport = sport
            st.session_state.athlete_info = {
                "name": athlete_name,
                "dob": dob,
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "resting_hr": resting_hr,
                "max_hr": max_hr,
                "test_date": test_date
            }
    else:
        st.info("Please enter test data in the 'Data Input' tab")
        
# Tab 3: Export Report
with tab3:
    st.header("Generate Report")
    
    if 'results' in st.session_state:
        st.subheader("Report Settings")
        
        include_logo = st.checkbox("Include Company Logo", value=True)
        include_training_zones = st.checkbox("Include Training Zones", value=True)
        include_raw_data = st.checkbox("Include Raw Test Data", value=False)
        
        additional_notes = st.text_area("Additional Notes", height=100)
        
        if st.button("Generate PDF Report"):
            try:
                with st.spinner("Generating PDF report..."):
                    report_pdf = generate_pdf_report(
                        st.session_state.athlete_info,
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
                report_filename = f"{st.session_state.athlete_info['name'].replace(' ', '_')}_{st.session_state.sport}_Threshold_Report.pdf"
                href = f'<a href="data:application/pdf;base64,{b64}" download="{report_filename}">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Preview note
                st.success("PDF generated successfully! Click the link above to download.")
                st.info("Note: PDF preview may not display correctly in Streamlit. Please download the file to view it.")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                st.info("Try disabling some options like 'Include Logo' and try again.")
    else:
        st.info("Complete the analysis in the 'Analysis & Results' tab to generate a report")

# Footer
st.markdown("""
<footer>
    <p>© Lindblom Coaching - Professional Threshold Analysis</p>
</footer>
""", unsafe_allow_html=True)
