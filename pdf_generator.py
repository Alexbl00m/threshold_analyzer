from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import PageBreak, Flowable
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from data_processing import estimate_vo2max
import os
import base64

# Brand colors
BRAND_COLORS = {
    'primary': '#E6754E',    # Orange
    'secondary': '#2E4057',  # Dark blue
    'accent1': '#48A9A6',    # Teal
    'accent2': '#D4B483',    # Light brown
    'accent3': '#C1666B',    # Red
    'background': '#F8F8F8', # Light gray
    'text': '#333333'        # Dark gray
}


class MCLine(Flowable):
    """Custom flowable for drawing horizontal lines"""
    
    def __init__(self, width, height=0, color=colors.HexColor('#E6754E')):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.height)
        self.canv.line(0, 0, self.width, 0)


def create_plot_for_pdf(data, results, sport, method_name=None):
    """
    Creates a simple matplotlib plot for the PDF report.
    
    Args:
        data: DataFrame with test data
        results: Dictionary of threshold results
        sport: Either "Cycling" or "Running"
        method_name: Optional specific method to highlight
        
    Returns:
        buf: BytesIO buffer containing the plot image
    """
    plt.figure(figsize=(8, 5))
    
    # Set up x column name and axis label
    if sport == "Cycling":
        x_col = "Power"
        x_label = "Power (Watts)"
    else:  # Running
        x_col = "Speed"
        x_label = "Speed (km/h)"
    
    # Sort the data by intensity
    sorted_data = data.sort_values(by=x_col)
    
    # Plot the lactate curve
    plt.scatter(sorted_data[x_col], sorted_data["Lactate"], 
                s=60, color=BRAND_COLORS['secondary'], label='Test Data')
    
    # Draw smooth curve through the data points
    for method, result in results.items():
        if 'curve_fit' in result['details']:
            curve_fit = result['details']['curve_fit']
            x_range = np.linspace(sorted_data[x_col].min(), sorted_data[x_col].max(), 1000)
            plt.plot(x_range, curve_fit(x_range), '--', linewidth=1, alpha=0.7,
                  color=BRAND_COLORS['primary'] if method == method_name else '#999999')
    
    # Plot the selected threshold
    if method_name and method_name in results:
        threshold = results[method_name]['threshold']
        plt.axvline(x=threshold, color=BRAND_COLORS['primary'], linestyle='--')
        
        # Get lactate at threshold
        curve_fit = results[method_name]['details'].get('curve_fit')
        if curve_fit:
            lactate_at_threshold = curve_fit(threshold)
            plt.scatter([threshold], [lactate_at_threshold], 
                        s=100, color=BRAND_COLORS['primary'], 
                        edgecolor='white', zorder=10)
            
            # Label the threshold point
            if sport == "Cycling":
                label = f"{method_name}: {threshold:.1f} W"
            else:
                pace_min = int(60 / threshold)
                pace_sec = int(60 * (60 / threshold - pace_min))
                label = f"{method_name}: {threshold:.1f} km/h ({pace_min}:{pace_sec:02d} min/km)"
                
            plt.annotate(label, xy=(threshold, lactate_at_threshold),
                         xytext=(threshold, lactate_at_threshold + 1),
                         ha='center', va='bottom',
                         color=BRAND_COLORS['primary'],
                         fontweight='bold')
    
    # Set up the plot
    plt.title('Lactate Response Curve', fontsize=14, color=BRAND_COLORS['primary'])
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel('Lactate (mmol/L)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    
    return buf


def generate_pdf_report(athlete_info, data, results, training_zones, sport, 
                        additional_notes=None, include_logo=True, 
                        include_training_zones=True, include_raw_data=False):
    """
    Generates a PDF report of the threshold test results.
    
    Args:
        athlete_info: Dictionary with athlete information
        data: DataFrame with test data
        results: Dictionary of threshold results
        training_zones: List of dictionaries with training zone information
        sport: Either "Cycling" or "Running"
        additional_notes: Optional additional notes to include
        include_logo: Whether to include the company logo
        include_training_zones: Whether to include training zones
        include_raw_data: Whether to include raw test data
        
    Returns:
        pdf_bytes: PDF report as bytes
    """
    buffer = BytesIO()
    
    # Set up the document
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles - use unique names to avoid conflicts
    if 'GoingLongTitle' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#E6754E'),
            alignment=TA_CENTER,
            spaceAfter=12
        ))
    
    if 'GoingLongSubtitle' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2E4057'),
            spaceAfter=10
        ))
    
    if 'GoingLongSectionTitle' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongSectionTitle',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#E6754E'),
            spaceAfter=8
        ))
    
    if 'GoingLongNormal' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        ))
    
    if 'GoingLongBold' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongBold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.HexColor('#333333')
        ))
    
    if 'GoingLongCenterText' not in styles:
        styles.add(ParagraphStyle(
            name='GoingLongCenterText',
            parent=styles['Normal'],
            alignment=TA_CENTER
        ))
    
    # Create story (list of flowables)
    story = []
    
    # Add logo if requested
    if include_logo:
        try:
            logo_path = os.path.join('assets', 'Logotype_Light@2x.png')
            if os.path.exists(logo_path):
                logo = Image(logo_path, width=2*inch, height=1*inch)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            # Skip logo if there's an error
            pass
    
    # Add title
    title = f"{sport} Threshold Analysis Report"
    story.append(Paragraph(title, styles['GoingLongTitle']))
    story.append(MCLine(450, 2))
    story.append(Spacer(1, 0.2*inch))
    
    # Add date
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Report Date: {date_str}", styles['GoingLongCenterText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Athlete information
    story.append(Paragraph("Athlete Information", styles['GoingLongSectionTitle']))
    
    athlete_data = []
    if athlete_info:
        name = athlete_info.get('name', 'N/A')
        dob = athlete_info.get('dob', None)
        age = athlete_info.get('age', 'N/A')
        gender = athlete_info.get('gender', 'N/A')
        height = athlete_info.get('height', 0)
        weight = athlete_info.get('weight', 0)
        resting_hr = athlete_info.get('resting_hr', 'N/A')
        max_hr = athlete_info.get('max_hr', 'N/A')
        test_date = athlete_info.get('test_date', None)
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2) if height > 0 and weight > 0 else 0
        
        athlete_data = [
            ["Name:", name],
            ["Date of Birth:", dob.strftime("%Y-%m-%d") if dob else "N/A"],
            ["Age:", f"{age} years" if age else "N/A"],
            ["Gender:", gender],
            ["Height:", f"{height} cm"],
            ["Weight:", f"{weight} kg"],
            ["BMI:", f"{bmi:.1f}"],
            ["Resting HR:", f"{resting_hr} bpm"],
            ["Maximum HR:", f"{max_hr} bpm"],
            ["Test Date:", test_date.strftime("%Y-%m-%d") if test_date else "N/A"]
        ]
    
    # Create table for athlete info
    if athlete_data:
        t = Table(athlete_data, colWidths=[100, 300])
        t.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2E4057')),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Threshold Results
    story.append(Paragraph("Threshold Analysis Results", styles['GoingLongSectionTitle']))
    
    # Choose primary method if available
    primary_method = None
    if "Modified Dmax" in results:
        primary_method = "Modified Dmax"
    elif results:
        primary_method = list(results.keys())[0]
    
    # Create plot with primary method highlighted
    if primary_method:
        plot_buf = create_plot_for_pdf(data, results, sport, primary_method)
        img = Image(plot_buf, width=6*inch, height=3.75*inch)
        story.append(img)
    
    story.append(Spacer(1, 0.2*inch))
    
    # Table of threshold values
    if results:
        threshold_data = [["Method", "Threshold", "Relative"]]
        
        for method_name, result in results.items():
            threshold = result['threshold']
            
            if sport == "Cycling":
                threshold_str = f"{threshold:.1f} W"
                relative_str = f"{threshold / athlete_info.get('weight', 1):.2f} W/kg"
            else:  # Running
                threshold_str = f"{threshold:.2f} km/h"
                pace_min = int(60 / threshold)
                pace_sec = int((60 / threshold - pace_min) * 60)
                pace_str = f"{pace_min}:{pace_sec:02d} min/km"
                relative_str = pace_str
            
            threshold_data.append([method_name, threshold_str, relative_str])
        
        t = Table(threshold_data, colWidths=[180, 120, 120])
        t.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            # Highlight the primary method row
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F8F8F8') if len(threshold_data) > 1 else None),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#E6754E') if len(threshold_data) > 1 else None),
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Estimated VO2max
    if primary_method and 'weight' in athlete_info and athlete_info['weight'] > 0:
        primary_threshold = results[primary_method]['threshold']
        vo2max = estimate_vo2max(
            primary_threshold, 
            athlete_info['weight'], 
            sport, 
            athlete_info.get('gender', 'Male')
        )
        
        story.append(Paragraph("Physiological Parameters", styles['GoingLongSectionTitle']))
        
        phys_data = [
            ["Estimated VO2max:", f"{vo2max:.1f} ml/kg/min"],
        ]
        
        # Add heart rate at threshold if available
        if 'hr_at_threshold' in results[primary_method]['details'] and results[primary_method]['details']['hr_at_threshold'] is not None:
            hr_at_threshold = results[primary_method]['details']['hr_at_threshold']
            phys_data.append(["Heart Rate at Threshold:", f"{int(hr_at_threshold)} bpm"])
            
            # Calculate as % of max
            if max_hr and max_hr > 0:
                hr_percent = (hr_at_threshold / max_hr) * 100
                phys_data.append(["Threshold as % of Max HR:", f"{hr_percent:.1f}%"])
        
        t = Table(phys_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2E4057')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Training Zones
    if include_training_zones and training_zones:
        story.append(Paragraph("Training Zones", styles['GoingLongSectionTitle']))
        
        # Column headers depend on sport
        if sport == "Cycling":
            zone_headers = ["Zone", "Power Range", "Heart Rate", "Description"]
            col_widths = [100, 100, 100, 150]
        else:  # Running
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
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        
        # Alternate row colors for better readability
        for i in range(1, len(zone_data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8F8F8')))
        
        t.setStyle(TableStyle(table_style))
        story.append(t)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Additional notes
    if additional_notes:
        story.append(Paragraph("Additional Notes", styles['GoingLongSectionTitle']))
        story.append(Paragraph(additional_notes, styles['GoingLongNormal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Raw test data
    if include_raw_data and not data.empty:
        story.append(Paragraph("Raw Test Data", styles['GoingLongSectionTitle']))
        
        # Use only relevant columns
        if sport == "Cycling":
            display_cols = ["Power", "HeartRate", "Lactate", "RPE"]
        else:  # Running
            display_cols = ["Speed", "Pace", "HeartRate", "Lactate", "RPE"]
        
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
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]
        
        # Alternate row colors
        for i in range(1, len(raw_data)):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8F8F8')))
        
        t.setStyle(TableStyle(table_style))
        story.append(t)
    
    # Footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor(BRAND_COLORS['secondary']))
        footer_text = f"© Lindblom Coaching - Professional Threshold Analysis"
        canvas.drawCentredString(doc.width / 2 + doc.leftMargin, 0.5 * inch, footer_text)
        canvas.restoreState()
    
    # Build document
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    
    # Get PDF as bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes