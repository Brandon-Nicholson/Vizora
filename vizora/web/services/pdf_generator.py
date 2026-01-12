"""
PDF Report Generator

Generates professional PDF reports from analysis results using ReportLab.
"""

import io
import base64
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage


class PDFReportGenerator:
    """
    Generates PDF reports from Vizora analysis results.
    """

    # Color scheme matching the Vizora brand
    COLORS = {
        'primary': colors.HexColor('#0077b6'),
        'secondary': colors.HexColor('#5a2ea6'),
        'dark': colors.HexColor('#0a0a0f'),
        'card': colors.HexColor('#f5f7fb'),
        'border': colors.HexColor('#d6dbe5'),
        'text': colors.HexColor('#1f2430'),
        'text_muted': colors.HexColor('#5f6b7a'),
    }

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configure custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=20,
            textColor=self.COLORS['text'],
            alignment=TA_CENTER,
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=self.COLORS['text_muted'],
            alignment=TA_CENTER,
            spaceAfter=30,
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=self.COLORS['primary'],
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='VizoraBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.COLORS['text'],
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14,
        ))

        # Bullet style
        self.styles.add(ParagraphStyle(
            name='BulletText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.COLORS['text'],
            leftIndent=20,
            spaceAfter=4,
            bulletIndent=10,
        ))

    def generate(
        self,
        figures: list[dict],
        metrics: Optional[dict],
        summary_markdown: str,
        plan: dict,
        metadata: dict,
    ) -> bytes:
        """
        Generate a PDF report from analysis results.

        Args:
            figures: List of figure dicts with 'type', 'name', 'base64_png'
            metrics: Model metrics dict (model_name -> metrics)
            summary_markdown: AI-generated summary in markdown
            plan: Execution plan dict
            metadata: Analysis metadata (goal, mode, dataset info)

        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []

        # Cover Page
        story.extend(self._build_cover_page(metadata))
        story.append(PageBreak())

        # Executive Summary
        story.extend(self._build_summary_section(summary_markdown))
        story.append(PageBreak())

        # Visualizations
        if figures:
            story.extend(self._build_figures_section(figures))
            story.append(PageBreak())

        # Model Performance
        if metrics:
            story.extend(self._build_metrics_section(metrics))
            story.append(PageBreak())

        # Execution Plan Summary
        story.extend(self._build_plan_section(plan))

        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _build_cover_page(self, metadata: dict) -> list:
        """Build the cover page."""
        elements = []

        # Add spacer to center content vertically
        elements.append(Spacer(1, 2 * inch))

        # Title
        mode = metadata.get('mode', 'Analysis').upper()
        title = f"Vizora {mode} Report"
        elements.append(Paragraph(title, self.styles['ReportTitle']))

        # Goal
        goal = metadata.get('goal', 'Data Analysis')
        elements.append(Paragraph(f'"{goal}"', self.styles['ReportSubtitle']))

        elements.append(Spacer(1, 0.5 * inch))

        # Metadata table
        dataset_info = metadata.get('dataset_info', {})
        meta_data = [
            ['Generated', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Dataset', f"{dataset_info.get('rows', 'N/A')} rows x {dataset_info.get('columns', 'N/A')} columns"],
            ['Analysis Mode', mode.title()],
        ]

        if dataset_info.get('target'):
            meta_data.append(['Target Column', dataset_info.get('target')])

        meta_table = Table(meta_data, colWidths=[1.5 * inch, 4 * inch])
        meta_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), self.COLORS['text_muted']),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(meta_table)

        elements.append(Spacer(1, 2 * inch))

        # Footer
        elements.append(Paragraph(
            'Powered by Vizora - AI-Powered Data Analysis',
            ParagraphStyle(
                name='CoverFooter',
                fontSize=9,
                textColor=self.COLORS['text_muted'],
                alignment=TA_CENTER,
            )
        ))

        return elements

    def _build_summary_section(self, summary_markdown: str) -> list:
        """Build the executive summary section."""
        elements = []

        elements.append(Paragraph('Executive Summary', self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        # Convert markdown to paragraphs (simplified conversion)
        paragraphs = self._markdown_to_paragraphs(summary_markdown)
        elements.extend(paragraphs)

        return elements

    def _markdown_to_paragraphs(self, markdown: str) -> list:
        """Convert markdown text to ReportLab paragraphs (simplified)."""
        elements = []
        lines = markdown.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 0.1 * inch))
                continue

            # Headers
            if line.startswith('## '):
                elements.append(Paragraph(
                    line[3:],
                    self.styles['SectionHeader']
                ))
            elif line.startswith('# '):
                elements.append(Paragraph(
                    line[2:],
                    self.styles['SectionHeader']
                ))
            # Bullet points
            elif line.startswith('- ') or line.startswith('* '):
                # Clean up markdown formatting
                text = line[2:]
                text = text.replace('**', '')
                text = text.replace('*', '')
                text = text.replace('`', '')
                elements.append(Paragraph(
                    f"• {text}",
                    self.styles['BulletText']
                ))
            # Regular text
            else:
                # Clean up markdown formatting
                text = line.replace('**', '')
                text = text.replace('*', '')
                text = text.replace('`', '')
                if text:
                    elements.append(Paragraph(text, self.styles['VizoraBodyText']))

        return elements

    def _build_figures_section(self, figures: list[dict]) -> list:
        """Build the visualizations section."""
        elements = []

        elements.append(Paragraph('Visualizations', self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        # Group figures, 2 per page
        for i, fig in enumerate(figures):
            if i > 0 and i % 2 == 0:
                elements.append(PageBreak())

            # Figure title
            fig_title = fig.get('name', f"Figure {i + 1}")
            fig_type = fig.get('type', 'visualization')
            elements.append(Paragraph(
                f"{fig_title} ({fig_type})",
                ParagraphStyle(
                    name='FigureTitle',
                    fontSize=11,
                    textColor=self.COLORS['text'],
                    spaceAfter=8,
                )
            ))

            # Convert base64 to image
            try:
                img_data = self._decode_base64_image(fig.get('base64_png', ''))
                if img_data:
                    img = Image(img_data, width=6 * inch, height=4 * inch)
                    img.hAlign = 'CENTER'
                    elements.append(img)
            except Exception as e:
                elements.append(Paragraph(
                    f"[Image could not be rendered: {str(e)}]",
                    self.styles['VizoraBodyText']
                ))

            elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _decode_base64_image(self, base64_str: str) -> Optional[io.BytesIO]:
        """Decode a base64 image string to a BytesIO object."""
        if not base64_str:
            return None

        # Remove data URL prefix if present
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',', 1)[1]

        try:
            img_data = base64.b64decode(base64_str)
            return io.BytesIO(img_data)
        except Exception:
            return None

    def _build_metrics_section(self, metrics: dict) -> list:
        """Build the model performance section."""
        elements = []

        elements.append(Paragraph('Model Performance', self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        if not metrics:
            elements.append(Paragraph(
                'No model metrics available.',
                self.styles['VizoraBodyText']
            ))
            return elements

        # Build metrics table
        # Get all unique metric names
        all_metrics = set()
        for model_metrics in metrics.values():
            all_metrics.update(model_metrics.keys())

        # Filter to important metrics
        priority_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
            'mse', 'rmse', 'mae', 'r2',
            'cv_accuracy_mean', 'brier_score'
        ]
        ordered_metrics = [m for m in priority_metrics if m in all_metrics]

        # Build table data
        headers = ['Metric'] + list(metrics.keys())
        table_data = [headers]

        for metric_name in ordered_metrics:
            row = [metric_name.replace('_', ' ').title()]
            for model_name in metrics.keys():
                value = metrics[model_name].get(metric_name)
                if value is not None:
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                else:
                    row.append('-')
            table_data.append(row)

        # Create table
        col_widths = [1.8 * inch] + [1.5 * inch] * len(metrics)
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            # Header style
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['card']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.COLORS['primary']),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            # Body style
            ('TEXTCOLOR', (0, 1), (-1, -1), self.COLORS['text']),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['border']),
            # Alignment
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.COLORS['dark'], self.COLORS['card']]),
        ]))

        elements.append(table)

        return elements

    def _build_plan_section(self, plan: dict) -> list:
        """Build the execution plan summary section."""
        elements = []

        elements.append(Paragraph('Execution Plan', self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        # Summarize each section of the plan
        sections = ['cleaning', 'eda', 'preprocessing', 'modeling', 'evaluation', 'analysis']

        for section in sections:
            actions = plan.get(section, [])
            if not actions:
                continue

            elements.append(Paragraph(
                section.title(),
                ParagraphStyle(
                    name='PlanSection',
                    fontSize=11,
                    textColor=self.COLORS['text'],
                    spaceBefore=10,
                    spaceAfter=5,
                    fontName='Helvetica-Bold',
                )
            ))

            for action in actions:
                action_type = action.get('action', 'unknown')
                reason = action.get('reason', '')
                elements.append(Paragraph(
                    f"• {action_type}: {reason}",
                    self.styles['BulletText']
                ))

        # Notes
        notes = plan.get('notes', [])
        if notes:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph('Notes', ParagraphStyle(
                name='NotesHeader',
                fontSize=11,
                textColor=self.COLORS['text'],
                fontName='Helvetica-Bold',
            )))
            for note in notes:
                elements.append(Paragraph(f"• {note}", self.styles['BulletText']))

        return elements


# Singleton instance
pdf_generator = PDFReportGenerator()
