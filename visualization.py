import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Use Going Long brand colors
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


def create_lactate_curve_plot(intensity_values, lactate_values, heart_rate_values, results, x_label="Power (Watts)", sport="Cycling"):
    """
    Creates a matplotlib plot showing the lactate curve with threshold points.
    
    Args:
        intensity_values: Array of power or speed values
        lactate_values: Array of lactate values
        heart_rate_values: Array of heart rate values (can be None)
        results: Dictionary of threshold results from different methods
        x_label: Label for x-axis
        sport: Either "Cycling" or "Running"
        
    Returns:
        fig: Matplotlib figure object
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Sort the data by intensity
    sort_idx = np.argsort(intensity_values)
    intensity_sorted = intensity_values[sort_idx]
    lactate_sorted = lactate_values[sort_idx]
    
    # Ensure heart rate values match the sorted intensity
    if heart_rate_values is not None and len(heart_rate_values) == len(intensity_values):
        hr_sorted = heart_rate_values[sort_idx]
    else:
        hr_sorted = np.zeros_like(intensity_sorted)
    
    # Plot the lactate curve
    ax1.scatter(intensity_sorted, lactate_sorted, s=80, color=BRAND_COLORS['secondary'], label='Test Data')
    
    # Create a smooth line through the data points
    for method_name, result in results.items():
        if 'curve_fit' in result['details']:
            curve_fit = result['details']['curve_fit']
            dense_x = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
            dense_y = curve_fit(dense_x)
            ax1.plot(dense_x, dense_y, '--', linewidth=1, alpha=0.7, 
                     color=THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text']))
    
    # Plot each threshold
    for i, (method_name, result) in enumerate(results.items()):
        threshold = result['threshold']
        color = THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text'])
        
        # Plot vertical line at threshold
        ax1.axvline(x=threshold, color=color, linestyle='--', alpha=0.7)
        
        # Plot the threshold point on the lactate curve
        if 'curve_fit' in result['details']:
            curve_fit = result['details']['curve_fit']
            lactate_at_threshold = curve_fit(threshold)
            ax1.scatter([threshold], [lactate_at_threshold], marker='o', s=100, 
                        color=color, edgecolor='white', zorder=10)
        
        # Annotate with method name
        if sport == "Cycling":
            label = f"{method_name}: {threshold:.1f} W"
        else:
            pace_min = int(60 / threshold)
            pace_sec = int(60 * (60 / threshold - pace_min))
            label = f"{method_name}: {threshold:.1f} km/h ({pace_min}:{pace_sec:02d} min/km)"
        
        ax1.annotate(label, xy=(threshold, 0), xytext=(threshold, -0.5 - i*0.3),
                     arrowprops=dict(arrowstyle='->', color=color),
                     va='top', ha='center', color=color)
    
    # Set up the first subplot (lactate curve)
    ax1.set_title('Lactate Response Curve', fontsize=16, color=BRAND_COLORS['primary'])
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Lactate (mmol/L)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    # Plot the heart rate
    ax2.scatter(intensity_sorted, hr_sorted, s=80, color=BRAND_COLORS['accent3'], label='Heart Rate')
    
    # Fit a linear or polynomial model to heart rate
    if len(intensity_sorted) >= 3:
        try:
            hr_fit = np.polyfit(intensity_sorted, hr_sorted, 1)
            hr_line = np.poly1d(hr_fit)
            dense_x = np.linspace(intensity_sorted.min(), intensity_sorted.max(), 1000)
            ax2.plot(dense_x, hr_line(dense_x), '--', color=BRAND_COLORS['accent3'], alpha=0.7)
        except:
            pass  # Skip fitting if it fails
    
    # Plot each threshold on the heart rate subplot
    for method_name, result in results.items():
        threshold = result['threshold']
        color = THRESHOLD_COLORS.get(method_name, BRAND_COLORS['text'])
        
        # Plot vertical line at threshold
        ax2.axvline(x=threshold, color=color, linestyle='--', alpha=0.7)
        
        # If we have a heart rate model, plot the HR at threshold
        if len(intensity_sorted) >= 3:
            try:
                hr_at_threshold = hr_line(threshold)
                ax2.scatter([threshold], [hr_at_threshold], marker='o', s=100, 
                            color=color, edgecolor='white', zorder=10)
                
                # Store HR at threshold in results
                results[method_name]['details']['hr_at_threshold'] = hr_at_threshold
            except:
                pass
    
    # Set up the second subplot (heart rate)
    ax2.set_title('Heart Rate Response', fontsize=16, color=BRAND_COLORS['primary'])
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set a reasonable lower limit for heart rate
    min_hr = max(60, np.min(hr_sorted) - 10)
    ax2.set_ylim(bottom=min_hr)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Add watermark
    fig.text(0.5, 0.01, "© Lindblom Coaching", ha='center', 
             color=BRAND_COLORS['secondary'], alpha=0.7, fontsize=8)
    
    return fig


def create_interactive_plot(data, results, x_column="Power", sport="Cycling"):
    """
    Creates an interactive Plotly plot showing the lactate curve and heart rate response.
    
    Args:
        data: DataFrame with test data
        results: Dictionary of threshold results from different methods
        x_column: Column name for intensity values (Power or Speed)
        sport: Either "Cycling" or "Running"
        
    Returns:
        fig: Plotly figure object
    """
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Lactate Response Curve', 'Heart Rate Response')
    )
    
    # Add lactate scatter points
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
    
    # Add heart rate scatter points
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
            annotation_text=method_name,
            annotation_position="top right",
        )
        
        # Add threshold points on the lactate curve
        if 'curve_fit' in result['details']:
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
                    font=dict(color=color)
                )
            except:
                pass
        
        # Add threshold points on the heart rate curve
        if 'hr_at_threshold' in result['details'] and result['details']['hr_at_threshold'] is not None:
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
                font=dict(color=color)
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
    
    # Add copyright
    fig.add_annotation(
        text="© Lindblom Coaching",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(color=BRAND_COLORS['secondary'], size=10)
    )
    
    return fig


def create_zone_color_scale(num_zones=5):
    """
    Creates a color scale for training zones.
    
    Args:
        num_zones: Number of training zones
        
    Returns:
        zone_colors: List of colors for each zone
    """
    base_colors = [
        BRAND_COLORS['accent1'],  # Zone 1 - Recovery
        BRAND_COLORS['accent2'],  # Zone 2 - Endurance
        BRAND_COLORS['primary'],  # Zone 3 - Tempo
        BRAND_COLORS['accent3'],  # Zone 4 - Threshold
        BRAND_COLORS['secondary'] # Zone 5 - VO2max
    ]
    
    # If more zones needed, cycle through the colors
    return base_colors[:num_zones]