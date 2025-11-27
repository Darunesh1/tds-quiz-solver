"""
Visualization primitives: charts, plots, and graphs.
Generates base64-encoded images for submission (<1MB constraint).
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.logger import setup_logger
from app.utils.exceptions import AnalysisError
from app.utils.helpers import get_file_size_mb

logger = setup_logger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100


class Visualizer:
    """
    Chart generation with automatic base64 encoding.
    Ensures images are <1MB for submission payloads.
    """

    def __init__(self, max_size_mb: float = 0.5):
        """
        Initialize visualizer.

        Args:
            max_size_mb: Maximum image size in MB (default 0.5MB for safety)
        """
        self.max_size_mb = max_size_mb

    def _encode_figure(self, fig: plt.Figure, format: str = "png") -> str:
        """
        Convert matplotlib figure to base64 string.

        Args:
            fig: Matplotlib figure
            format: Image format (png, jpg)

        Returns:
            Base64 data URI string
        """
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches="tight", pad_inches=0.1)
            buffer.seek(0)

            # Check size
            size_mb = len(buffer.getvalue()) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                logger.warning(
                    f"⚠️ Image size {size_mb:.2f}MB exceeds {self.max_size_mb}MB, compressing..."
                )
                # Retry with JPEG and lower quality
                buffer = BytesIO()
                fig.savefig(
                    buffer,
                    format="jpg",
                    bbox_inches="tight",
                    pad_inches=0.1,
                    quality=70,
                )
                buffer.seek(0)
                size_mb = len(buffer.getvalue()) / (1024 * 1024)

            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mime_type = f"image/{format}"
            data_uri = f"data:{mime_type};base64,{image_base64}"

            logger.info(f"✅ Figure encoded: {size_mb:.2f}MB")
            plt.close(fig)
            return data_uri

        except Exception as e:
            plt.close(fig)
            raise AnalysisError(f"Figure encoding failed: {e}")

    def line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        title: str = "Line Chart",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> str:
        """
        Create line chart.

        Args:
            df: DataFrame
            x: X-axis column
            y: Y-axis column(s)
            title: Chart title
            xlabel, ylabel: Axis labels

        Returns:
            Base64 data URI
        """
        try:
            logger.info(f"📈 Creating line chart: {y} vs {x}")

            fig, ax = plt.subplots(figsize=(10, 6))

            if isinstance(y, str):
                y = [y]

            for col in y:
                ax.plot(df[x], df[col], marker="o", label=col)

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(xlabel or x)
            ax.set_ylabel(ylabel or ", ".join(y))
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._encode_figure(fig)

        except Exception as e:
            logger.error(f"❌ Line chart error: {e}")
            raise AnalysisError(f"Line chart creation failed: {e}")

    def bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Bar Chart",
        horizontal: bool = False,
    ) -> str:
        """
        Create bar chart.

        Args:
            df: DataFrame
            x: Category column
            y: Value column
            title: Chart title
            horizontal: Horizontal bars if True

        Returns:
            Base64 data URI
        """
        try:
            logger.info(f"📊 Creating bar chart: {y} by {x}")

            fig, ax = plt.subplots(figsize=(10, 6))

            if horizontal:
                ax.barh(df[x], df[y], color=sns.color_palette("husl", len(df)))
                ax.set_xlabel(y)
                ax.set_ylabel(x)
            else:
                ax.bar(df[x], df[y], color=sns.color_palette("husl", len(df)))
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            return self._encode_figure(fig)

        except Exception as e:
            logger.error(f"❌ Bar chart error: {e}")
            raise AnalysisError(f"Bar chart creation failed: {e}")

    def scatter_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Scatter Plot",
        hue: Optional[str] = None,
    ) -> str:
        """
        Create scatter plot.

        Args:
            df: DataFrame
            x: X-axis column
            y: Y-axis column
            title: Chart title
            hue: Color grouping column

        Returns:
            Base64 data URI
        """
        try:
            logger.info(f"📊 Creating scatter plot: {y} vs {x}")

            fig, ax = plt.subplots(figsize=(10, 6))

            if hue:
                for category in df[hue].unique():
                    mask = df[hue] == category
                    ax.scatter(df[mask][x], df[mask][y], label=category, alpha=0.6)
                ax.legend()
            else:
                ax.scatter(df[x], df[y], alpha=0.6)

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._encode_figure(fig)

        except Exception as e:
            logger.error(f"❌ Scatter plot error: {e}")
            raise AnalysisError(f"Scatter plot creation failed: {e}")

    def histogram(
        self, df: pd.DataFrame, column: str, bins: int = 30, title: str = "Histogram"
    ) -> str:
        """
        Create histogram.

        Args:
            df: DataFrame
            column: Column to plot
            bins: Number of bins
            title: Chart title

        Returns:
            Base64 data URI
        """
        try:
            logger.info(f"📊 Creating histogram: {column}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            return self._encode_figure(fig)

        except Exception as e:
            logger.error(f"❌ Histogram error: {e}")
            raise AnalysisError(f"Histogram creation failed: {e}")

    def heatmap(self, df: pd.DataFrame, title: str = "Correlation Heatmap") -> str:
        """
        Create correlation heatmap.

        Args:
            df: DataFrame (will calculate correlation)
            title: Chart title

        Returns:
            Base64 data URI
        """
        try:
            logger.info("📊 Creating heatmap")

            # Calculate correlation for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title(title, fontsize=14, fontweight="bold")

            plt.tight_layout()
            return self._encode_figure(fig)

        except Exception as e:
            logger.error(f"❌ Heatmap error: {e}")
            raise AnalysisError(f"Heatmap creation failed: {e}")


# Global visualizer instance
visualizer = Visualizer()
