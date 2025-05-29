"""PDF Dashboard Generator based on the Portfolio Optimization Results.

This module creates comprehensive PDF reports from portfolio optimization results.
"""

import tempfile
import shutil
import datetime
import os
from contextlib import contextmanager
from io import BytesIO
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@contextmanager
def temporary_directory():
    """Context manager for temporary directory that ensures cleanup."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


class PDFDashboardGenerator:
    """Generate comprehensive PDF dashboards from portfolio optimization results.
    
    Focused on generating PDF bytes for cloud storage, with optional disk saving.
    """
    
    def __init__(self, 
                 title: str = "Portfolio Optimization Dashboard",
                 subtitle: str = "Investment Analysis Report",
                 company_name: str = "Vibe Trader"):
        """Initialize the PDF dashboard generator.
        
        Args:
            title: Main title for the dashboard
            subtitle: Subtitle for the dashboard
            company_name: Company/organization name
        """
        self.title = title
        self.subtitle = subtitle
        self.company_name = company_name
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles = getSampleStyleSheet()
        
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f77b4')
        )
        
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
        
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1f77b4')
        )

    def _validate_results(self, results: Dict) -> None:
        """Validate that results dictionary has required structure."""
        required_keys = ['inputs', 'results']
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key '{key}' in results dictionary")
        
        # Check for essential nested keys
        required_inputs = ['tickers', 'start_portfolio', 'target_portfolio', 'horizon_years']
        required_results = ['weights', 'success_prob', 'avg_final', 'volatility']
        
        for key in required_inputs:
            if key not in results['inputs']:
                raise ValueError(f"Missing required input key '{key}'")
                
        for key in required_results:
            if key not in results['results']:
                raise ValueError(f"Missing required result key '{key}'")

    def _create_charts(self, results: Dict, temp_dir: str) -> Dict[str, str]:
        """Create enhanced charts in temporary directory."""
        chart_paths = {}
        
        try:
            # Set professional style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # 1. Executive Summary Chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Portfolio Optimization Summary', fontsize=16, fontweight='bold')
            
            # Success probability gauge
            success_prob = results['results']['success_prob']
            ax1.pie([success_prob, 1-success_prob], labels=['Success', 'Risk'], 
                    colors=['#2ca02c', '#d62728'], startangle=90, autopct='%1.1f%%')
            ax1.set_title(f'Success Probability\n{success_prob:.1%}', fontweight='bold')
            
            # Risk metrics comparison
            metrics = ['Volatility', 'Max Drawdown', 'Worst Day']
            values = [
                results['results'].get('volatility', 0), 
                results['results'].get('avg_drawdown', 0),
                results['results'].get('avg_worst_day', 0)
            ]
            limits = [
                results['inputs'].get('sigma_max', 0.2),
                results['inputs'].get('max_drawdown', 0.1), 
                results['inputs'].get('worst_day_limit', 0.05)
            ]
            
            x_pos = np.arange(len(metrics))
            ax2.bar(x_pos, values, color=['#ff7f0e', '#d62728', '#9467bd'], alpha=0.7)
            ax2.plot(x_pos, limits, 'ro-', linewidth=2, markersize=8, label='Limits')
            ax2.set_xlabel('Risk Metrics')
            ax2.set_ylabel('Value')
            ax2.set_title('Risk Profile vs Constraints', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(metrics, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Portfolio value projection
            start_val = results['inputs']['start_portfolio']
            target_val = results['inputs']['target_portfolio']
            avg_final = results['results']['avg_final']
            
            categories = ['Start', 'Target', 'Expected']
            values_proj = [start_val, target_val, avg_final]
            colors_proj = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            ax3.bar(categories, values_proj, color=colors_proj, alpha=0.7)
            ax3.set_ylabel('Portfolio Value ($)')
            ax3.set_title('Value Projection', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Asset allocation pie chart
            weights = results['results']['weights']
            tickers = results['inputs']['tickers']
            
            # Only show assets with >1% allocation
            significant_weights = [(w, t) for w, t in zip(weights, tickers) if w > 0.01]
            if len(significant_weights) < len(weights):
                other_weight = sum(w for w, t in zip(weights, tickers) if w <= 0.01)
                if other_weight > 0:
                    significant_weights.append((other_weight, 'Other'))
            
            if significant_weights:
                weights_sig, tickers_sig = zip(*significant_weights)
                ax4.pie(weights_sig, labels=tickers_sig, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Asset Allocation', fontweight='bold')
            
            plt.tight_layout()
            chart_path = os.path.join(temp_dir, 'executive_summary.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            chart_paths['executive_summary'] = chart_path
            
            # 2. Asset Allocation Details
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Horizontal bar chart
            y_pos = np.arange(len(tickers))
            bars = ax1.barh(y_pos, weights, color=sns.color_palette("husl", len(tickers)))
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(tickers)
            ax1.set_xlabel('Weight')
            ax1.set_title('Portfolio Weights by Asset', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                ax1.text(weight + 0.01, i, f'{weight:.1%}', va='center')
            
            # Risk contribution (simplified)
            vol = results['results'].get('volatility', 0.1)
            risk_contrib = [w * vol for w in weights]
            
            bars2 = ax2.barh(y_pos, risk_contrib, color=sns.color_palette("viridis", len(tickers)))
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(tickers)
            ax2.set_xlabel('Risk Contribution')
            ax2.set_title('Risk Contribution by Asset', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(temp_dir, 'asset_allocation.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            chart_paths['asset_allocation'] = chart_path
            
        except Exception as e:
            # If chart creation fails, log but don't crash
            print(f"Warning: Chart creation failed: {e}")
            # Close any open figures
            plt.close('all')
            
        return chart_paths

    def _create_summary_table(self, results: Dict) -> Table:
        """Create a summary table of key metrics."""
        data = [['Metric', 'Value', 'Target/Limit', 'Status']]
        
        # Helper function to safely get values with defaults
        def safe_get(d, keys, default=0):
            try:
                result = d
                for key in keys if isinstance(keys, list) else [keys]:
                    result = result[key]
                return result
            except (KeyError, TypeError):
                return default
        
        # Build table data with safe access
        metrics = [
            ('Success Probability', 
             f"{safe_get(results, ['results', 'success_prob']):.1%}", 
             "Maximize", '✓'),
            ('Expected Final Value', 
             f"${safe_get(results, ['results', 'avg_final']):,.0f}", 
             f"${safe_get(results, ['inputs', 'target_portfolio']):,.0f}", 
             '✓' if safe_get(results, ['results', 'avg_final']) >= safe_get(results, ['inputs', 'target_portfolio']) else '⚠'),
            ('Portfolio Volatility', 
             f"{safe_get(results, ['results', 'volatility']):.1%}", 
             f"≤ {safe_get(results, ['inputs', 'sigma_max'], 0.2):.1%}", 
             '✓' if safe_get(results, ['results', 'volatility']) <= safe_get(results, ['inputs', 'sigma_max'], 0.2) else '⚠'),
        ]
        
        # Add optional metrics if they exist
        if safe_get(results, ['results', 'avg_drawdown']) > 0:
            metrics.append(('Average Drawdown', 
                          f"{safe_get(results, ['results', 'avg_drawdown']):.1%}", 
                          f"≤ {safe_get(results, ['inputs', 'max_drawdown'], 0.1):.1%}", 
                          '✓' if safe_get(results, ['results', 'avg_drawdown']) <= safe_get(results, ['inputs', 'max_drawdown'], 0.1) else '⚠'))
        
        data.extend(metrics)
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        return table

    def _create_allocation_table(self, results: Dict) -> Table:
        """Create asset allocation table."""
        weights = results['results']['weights']
        tickers = results['inputs']['tickers']
        total_value = results['inputs']['start_portfolio']
        
        data = [['Asset', 'Weight', 'Allocation ($)']]
        
        for ticker, weight in zip(tickers, weights):
            allocation = total_value * weight
            data.append([ticker, f"{weight:.1%}", f"${allocation:,.0f}"])
        
        # Add total row
        data.append(['TOTAL', '100.0%', f"${total_value:,.0f}"])
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        
        return table

    def generate_pdf(self, 
                    results: Dict, 
                    output_path: Optional[str] = None,
                    include_charts: bool = True) -> Union[bytes, str]:
        """Generate PDF dashboard.
        
        Args:
            results: Optimization results dictionary
            output_path: If provided, save to this path and return path. If None, return bytes.
            include_charts: Whether to include visual charts
            
        Returns:
            PDF bytes if output_path is None, file path string if output_path is provided
            
        Raises:
            ValueError: If results dictionary is invalid
        """
        # Validate inputs
        self._validate_results(results)
        
        # Create output directory if saving to disk
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use context manager for temporary directory
        with temporary_directory() as temp_dir:
            # Create charts if requested
            chart_paths = {}
            if include_charts:
                chart_paths = self._create_charts(results, temp_dir)
            
            # Create PDF document
            if output_path:
                doc = SimpleDocTemplate(output_path, pagesize=A4)
                buffer = None
            else:
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
            
            # Build content
            story = self._build_story(results, chart_paths)
            
            # Generate PDF
            doc.build(story)
            
            if output_path:
                return output_path
            else:
                pdf_bytes = buffer.getvalue()
                buffer.close()
                return pdf_bytes

    def _build_story(self, results: Dict, chart_paths: Dict[str, str]) -> list:
        """Build the PDF story content."""
        story = []
        
        # Title page
        story.append(Paragraph(self.title, self.title_style))
        story.append(Paragraph(self.subtitle, self.subtitle_style))
        story.append(Spacer(1, 30))
        
        # Report metadata
        report_date = datetime.datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Date:</b> {report_date}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Generated by:</b> {self.company_name}", self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.section_style))
        
        summary_text = f"""
        This portfolio optimization analysis was conducted for a {results['inputs']['horizon_years']}-year investment horizon 
        with an initial portfolio value of ${results['inputs']['start_portfolio']:,.0f} and a target value of 
        ${results['inputs']['target_portfolio']:,.0f}. The optimization achieved a success probability of 
        {results['results']['success_prob']:.1%} with an expected final portfolio value of 
        ${results['results']['avg_final']:,.0f}.
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Metrics Table
        story.append(Paragraph("Key Performance Metrics", self.section_style))
        story.append(self._create_summary_table(results))
        story.append(Spacer(1, 20))
        
        # Asset Allocation Table
        story.append(Paragraph("Asset Allocation", self.section_style))
        story.append(self._create_allocation_table(results))
        
        # Charts section
        if chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis", self.section_style))
            
            # Executive Summary Chart
            if 'executive_summary' in chart_paths:
                story.append(Paragraph("Portfolio Overview", self.styles['Heading3']))
                img = Image(chart_paths['executive_summary'], width=7*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Asset Allocation Chart
            if 'asset_allocation' in chart_paths:
                story.append(Paragraph("Asset Allocation Details", self.styles['Heading3']))
                img = Image(chart_paths['asset_allocation'], width=7*inch, height=3*inch)
                story.append(img)
        
        # Technical Details
        story.append(PageBreak())
        story.append(Paragraph("Technical Details", self.section_style))
        
        tech_details = f"""
        <b>Monte Carlo Scenarios:</b> {results['inputs'].get('scenarios', 'N/A'):,}<br/>
        <b>Optimization Time:</b> {results['results'].get('elapsed_time', 0):.2f} seconds<br/>
        <b>Actual Iterations:</b> {results['results'].get('iterations', 'N/A')}<br/>
        """
        
        story.append(Paragraph(tech_details, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Important Disclaimer", self.section_style))
        disclaimer_text = """
        This analysis is for informational purposes only and should not be considered as investment advice. 
        Past performance does not guarantee future results. All investments carry risk of loss. 
        Please consult with a qualified financial advisor before making investment decisions.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        return story


def generate_pdf_dashboard(inputs: Dict,
                          results: Dict, 
                          output_path: Optional[str] = None,
                          title: str = "Portfolio Optimization Dashboard",
                          subtitle: str = "Investment Analysis Report",
                          company_name: str = "Vibe Trader",
                          include_charts: bool = True) -> Union[bytes, str]:
    """Generate PDF dashboard - returns bytes or saves to file.
    
    Args:
        inputs: Input parameters dictionary containing tickers, portfolio values, constraints, etc.
        results: Results dictionary containing weights, success_prob, avg_final, volatility, etc.
        output_path: If provided, saves to file and returns path. If None, returns bytes.
        title: Dashboard title
        subtitle: Dashboard subtitle  
        company_name: Company name for branding
        include_charts: Whether to include visual charts
        
    Returns:
        PDF bytes if output_path is None, file path if output_path is provided
        
    Examples:
        # Get bytes for cloud upload
        pdf_bytes = generate_pdf_dashboard(inputs_dict, results_dict)
        
        # Save to file
        file_path = generate_pdf_dashboard(inputs_dict, results_dict, output_path="./report.pdf")
    """
    # Reconstruct the combined results dictionary for internal use
    results_dict = {
        'inputs': inputs,
        'results': results
    }
    
    generator = PDFDashboardGenerator(
        title=title,
        subtitle=subtitle, 
        company_name=company_name
    )
    
    return generator.generate_pdf(
        results=results_dict,
        output_path=output_path,
        include_charts=include_charts
    )


if __name__ == "__main__":
    # Example inputs and results structure (for testing)
    sample_inputs = {
        'tickers': ['SPY', 'QQQ', 'TLT', 'GLD', 'BIL'],
        'start_portfolio': 100000,
        'target_portfolio': 150000,
        'horizon_years': 5,
        'scenarios': 1000,
        'sigma_max': 0.15,
        'max_drawdown': 0.10,
    }
    
    sample_results = {
        'weights': [0.4, 0.3, 0.15, 0.1, 0.05],
        'success_prob': 0.75,
        'avg_final': 145000,
        'volatility': 0.12,
        'avg_drawdown': 0.08,
        'elapsed_time': 2.5,
        'iterations': 15,
    }
    
    # Generate PDF bytes (main use case)
    pdf_bytes = generate_pdf_dashboard(sample_inputs, sample_results)
    print(f"Generated PDF: {len(pdf_bytes):,} bytes")
    
    # Save to file (optional)
    file_path = generate_pdf_dashboard(sample_inputs, sample_results, output_path="./test_dashboard.pdf")
    print(f"Saved PDF to: {file_path}")
