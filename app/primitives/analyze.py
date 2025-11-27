"""
Data analysis primitives supporting 10x complexity scenarios.
Handles: statistics, filtering, aggregating, ML models, geospatial, network analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.logger import setup_logger
from app.utils.exceptions import AnalysisError

logger = setup_logger(__name__)


class DataAnalyzer:
    """
    Comprehensive data analysis toolkit.
    Supports statistics, transformations, ML, geospatial, and network analysis.
    """

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame: handle missing values, duplicates, data types.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"🧹 Cleaning DataFrame ({df.shape[0]} rows)")

            # Remove duplicate rows
            df = df.drop_duplicates()

            # Convert numeric columns with Indian format (1,23,456)
            for col in df.columns:
                if df[col].dtype == "object":
                    # Try converting to numeric if comma-separated
                    try:
                        df[col] = df[col].str.replace(",", "").astype(float)
                    except:
                        pass

            # Handle dates
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except:
                        pass

            logger.info(f"✅ Cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Cleaning error: {e}")
            raise AnalysisError(f"DataFrame cleaning failed: {e}")

    @staticmethod
    def basic_statistics(
        df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate basic statistics for numeric columns.

        Args:
            df: Input DataFrame
            columns: Specific columns (default: all numeric)

        Returns:
            Dictionary with statistics
        """
        try:
            logger.info("📊 Calculating basic statistics")

            if columns:
                df = df[columns]

            numeric_df = df.select_dtypes(include=[np.number])

            stats = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(numeric_df.columns),
                "summary": numeric_df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
            }

            logger.info(
                f"✅ Stats calculated for {len(numeric_df.columns)} numeric columns"
            )
            return stats

        except Exception as e:
            logger.error(f"❌ Statistics error: {e}")
            raise AnalysisError(f"Statistics calculation failed: {e}")

    @staticmethod
    def aggregate_data(
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        agg_funcs: Dict[str, Union[str, List[str]]],
    ) -> pd.DataFrame:
        """
        Aggregate data by groups.

        Args:
            df: Input DataFrame
            group_by: Column(s) to group by
            agg_funcs: Aggregation functions per column

        Returns:
            Aggregated DataFrame
        """
        try:
            logger.info(f"📊 Aggregating by: {group_by}")

            result = df.groupby(group_by).agg(agg_funcs).reset_index()

            logger.info(f"✅ Aggregated to {result.shape[0]} groups")
            return result

        except Exception as e:
            logger.error(f"❌ Aggregation error: {e}")
            raise AnalysisError(f"Aggregation failed: {e}")

    @staticmethod
    def filter_data(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter DataFrame based on conditions.

        Args:
            df: Input DataFrame
            conditions: Dict of column -> value/condition

        Returns:
            Filtered DataFrame
        """
        try:
            logger.info(f"🔍 Filtering with {len(conditions)} conditions")

            filtered = df.copy()
            for col, condition in conditions.items():
                if col not in filtered.columns:
                    continue

                if isinstance(condition, (list, tuple)):
                    # Filter by list of values
                    filtered = filtered[filtered[col].isin(condition)]
                elif isinstance(condition, dict):
                    # Complex condition (e.g., {"gt": 100, "lt": 500})
                    if "gt" in condition:
                        filtered = filtered[filtered[col] > condition["gt"]]
                    if "lt" in condition:
                        filtered = filtered[filtered[col] < condition["lt"]]
                    if "eq" in condition:
                        filtered = filtered[filtered[col] == condition["eq"]]
                else:
                    # Direct equality
                    filtered = filtered[filtered[col] == condition]

            logger.info(f"✅ Filtered to {filtered.shape[0]} rows")
            return filtered

        except Exception as e:
            logger.error(f"❌ Filtering error: {e}")
            raise AnalysisError(f"Filtering failed: {e}")

    @staticmethod
    def pivot_data(
        df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str = "sum"
    ) -> pd.DataFrame:
        """
        Create pivot table.

        Args:
            df: Input DataFrame
            index: Row labels
            columns: Column labels
            values: Values to aggregate
            aggfunc: Aggregation function

        Returns:
            Pivot table
        """
        try:
            logger.info(f"🔄 Creating pivot: {index} x {columns}")

            pivot = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=0,
            )

            logger.info(f"✅ Pivot created: {pivot.shape}")
            return pivot

        except Exception as e:
            logger.error(f"❌ Pivot error: {e}")
            raise AnalysisError(f"Pivot creation failed: {e}")

    @staticmethod
    def calculate_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.

        Args:
            df: Input DataFrame

        Returns:
            Correlation matrix
        """
        try:
            logger.info("📊 Calculating correlations")

            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()

            logger.info(f"✅ Correlation matrix: {corr.shape}")
            return corr

        except Exception as e:
            logger.error(f"❌ Correlation error: {e}")
            raise AnalysisError(f"Correlation calculation failed: {e}")

    @staticmethod
    def time_series_analysis(
        df: pd.DataFrame, date_column: str, value_column: str, freq: str = "D"
    ) -> Dict[str, Any]:
        """
        Basic time series analysis.

        Args:
            df: Input DataFrame
            date_column: Date column name
            value_column: Value column name
            freq: Resampling frequency (D=daily, W=weekly, M=monthly)

        Returns:
            Time series statistics
        """
        try:
            logger.info(f"📈 Time series analysis: {date_column} -> {value_column}")

            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column).sort_index()

            # Resample
            resampled = df[value_column].resample(freq).sum()

            # Calculate trend
            rolling_mean = resampled.rolling(window=7, min_periods=1).mean()

            result = {
                "total": float(resampled.sum()),
                "mean": float(resampled.mean()),
                "trend": rolling_mean.to_dict(),
                "period_start": str(resampled.index.min()),
                "period_end": str(resampled.index.max()),
                "data_points": len(resampled),
            }

            logger.info(f"✅ Time series analyzed: {result['data_points']} points")
            return result

        except Exception as e:
            logger.error(f"❌ Time series error: {e}")
            raise AnalysisError(f"Time series analysis failed: {e}")

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame, column: str, method: str = "iqr"
    ) -> pd.Series:
        """
        Detect outliers using IQR or Z-score method.

        Args:
            df: Input DataFrame
            column: Column to analyze
            method: 'iqr' or 'zscore'

        Returns:
            Boolean Series indicating outliers
        """
        try:
            logger.info(f"🔍 Detecting outliers in {column} using {method}")

            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df[column] < (Q1 - 1.5 * IQR)) | (
                    df[column] > (Q3 + 1.5 * IQR)
                )
            else:  # zscore
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > 3

            outlier_count = outliers.sum()
            logger.info(
                f"✅ Found {outlier_count} outliers ({outlier_count / len(df) * 100:.1f}%)"
            )
            return outliers

        except Exception as e:
            logger.error(f"❌ Outlier detection error: {e}")
            raise AnalysisError(f"Outlier detection failed: {e}")


# Global analyzer instance
data_analyzer = DataAnalyzer()
