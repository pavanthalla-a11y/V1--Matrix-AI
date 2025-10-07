import pandas as pd
import numpy as np
import psutil
from scipy import stats
from collections import Counter
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def analyze_data_distribution(synthetic_data: Dict[str, pd.DataFrame], seed_data: Dict[str, List[Dict[str, Any]]], metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive data distribution analysis comparing synthetic vs seed data.
    """
    logger.info("Starting comprehensive data distribution analysis...")
    
    analysis_report = {
        "generation_timestamp": datetime.now().isoformat(),
        "analysis_summary": {},
        "table_analyses": {},
        "statistical_tests": {},
        "data_quality_metrics": {},
        "synthesis_performance": {}
    }
    
    try:
        total_synthetic_records = sum(len(df) for df in synthetic_data.values())
        total_seed_records = sum(len(records) for records in seed_data.values())
        
        # Overall summary
        analysis_report["analysis_summary"] = {
            "total_tables": len(synthetic_data),
            "total_synthetic_records": total_synthetic_records,
            "total_seed_records": total_seed_records,
            "amplification_factor": round(total_synthetic_records / total_seed_records if total_seed_records > 0 else 0, 2),
            "memory_usage_mb": round(sum(df.memory_usage(deep=True).sum() for df in synthetic_data.values()) / (1024 * 1024), 2)
        }
        
        # Analyze each table
        for table_name, synthetic_df in synthetic_data.items():
            logger.info(f"Analyzing table: {table_name}")
            
            # Get corresponding seed data
            seed_records = seed_data.get(table_name, [])
            seed_df = pd.DataFrame.from_records(seed_records) if seed_records else pd.DataFrame()
            
            # Get table metadata - handle both dict and list formats
            tables_data = metadata_dict.get("tables", {})
            table_metadata = {}
            columns_metadata = {}
            
            if isinstance(tables_data, dict):
                table_metadata = tables_data.get(table_name, {})
                columns_metadata = table_metadata.get("columns", {})
            elif isinstance(tables_data, list):
                # Handle list format - find table by name
                for table_info in tables_data:
                    if isinstance(table_info, dict) and table_info.get("name") == table_name:
                        table_metadata = table_info
                        columns_metadata = table_info.get("columns", {})
                        break
            else:
                logger.warning(f"Unknown tables format in metadata: {type(tables_data)}")
            
            table_analysis = {
                "basic_stats": {
                    "synthetic_rows": len(synthetic_df),
                    "synthetic_columns": len(synthetic_df.columns),
                    "seed_rows": len(seed_df) if not seed_df.empty else 0,
                    "missing_values": synthetic_df.isnull().sum().to_dict(),
                    "duplicate_rows": synthetic_df.duplicated().sum(),
                    "memory_usage_kb": round(synthetic_df.memory_usage(deep=True).sum() / 1024, 2)
                },
                "column_distributions": {},
                "data_types_analysis": {},
                "uniqueness_analysis": {},
                "statistical_comparison": {}
            }
            
            # Analyze each column
            for col in synthetic_df.columns:
                col_metadata = columns_metadata.get(col, {})
                sdtype = col_metadata.get("sdtype", "unknown")
                
                synthetic_series = synthetic_df[col]
                seed_series = seed_df[col] if not seed_df.empty and col in seed_df.columns else pd.Series()
                
                col_analysis = {
                    "sdtype": sdtype,
                    "data_type": str(synthetic_series.dtype),
                    "unique_values": synthetic_series.nunique(),
                    "uniqueness_ratio": round(synthetic_series.nunique() / len(synthetic_series), 4),
                    "null_count": synthetic_series.isnull().sum(),
                    "null_percentage": round(synthetic_series.isnull().sum() / len(synthetic_series) * 100, 2)
                }
                
                # Type-specific analysis
                if sdtype in ['numerical', 'integer'] or synthetic_series.dtype in ['int64', 'float64']:
                    # Numerical analysis
                    col_analysis.update({
                        "min": float(synthetic_series.min()) if not synthetic_series.empty else None,
                        "max": float(synthetic_series.max()) if not synthetic_series.empty else None,
                        "mean": float(synthetic_series.mean()) if not synthetic_series.empty else None,
                        "median": float(synthetic_series.median()) if not synthetic_series.empty else None,
                        "std": float(synthetic_series.std()) if not synthetic_series.empty else None,
                        "skewness": float(stats.skew(synthetic_series.dropna())) if len(synthetic_series.dropna()) > 0 else None,
                        "kurtosis": float(stats.kurtosis(synthetic_series.dropna())) if len(synthetic_series.dropna()) > 0 else None,
                        "percentiles": {
                            "25th": float(synthetic_series.quantile(0.25)) if not synthetic_series.empty else None,
                            "50th": float(synthetic_series.quantile(0.5)) if not synthetic_series.empty else None,
                            "75th": float(synthetic_series.quantile(0.75)) if not synthetic_series.empty else None,
                            "90th": float(synthetic_series.quantile(0.9)) if not synthetic_series.empty else None,
                            "95th": float(synthetic_series.quantile(0.95)) if not synthetic_series.empty else None
                        }
                    })
                    
                    # Compare with seed data if available
                    if not seed_series.empty and len(seed_series.dropna()) > 0:
                        try:
                            # Statistical tests
                            ks_statistic, ks_p_value = stats.ks_2samp(synthetic_series.dropna(), seed_series.dropna())
                            col_analysis["statistical_tests"] = {
                                "kolmogorov_smirnov": {
                                    "statistic": float(ks_statistic),
                                    "p_value": float(ks_p_value),
                                    "interpretation": "Similar distributions" if ks_p_value > 0.05 else "Different distributions"
                                }
                            }
                            
                            # Distribution comparison
                            col_analysis["distribution_comparison"] = {
                                "seed_mean": float(seed_series.mean()),
                                "synthetic_mean": float(synthetic_series.mean()),
                                "mean_difference": float(abs(synthetic_series.mean() - seed_series.mean())),
                                "seed_std": float(seed_series.std()),
                                "synthetic_std": float(synthetic_series.std()),
                                "std_difference": float(abs(synthetic_series.std() - seed_series.std()))
                            }
                        except Exception as e:
                            logger.warning(f"Statistical comparison failed for column {col}: {e}")
                
                elif sdtype == 'categorical' or synthetic_series.dtype == 'object':
                    # Categorical analysis
                    value_counts = synthetic_series.value_counts()
                    col_analysis.update({
                        "top_categories": value_counts.head(10).to_dict(),
                        "category_count": len(value_counts),
                        "entropy": float(stats.entropy(value_counts.values)) if len(value_counts) > 0 else None,
                        "most_frequent": {
                            "value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                            "count": int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                            "frequency": round(value_counts.iloc[0] / len(synthetic_series), 4) if len(value_counts) > 0 else None
                        }
                    })
                    
                    # Compare with seed data
                    if not seed_series.empty:
                        seed_value_counts = seed_series.value_counts()
                        synthetic_categories = set(value_counts.index)
                        seed_categories = set(seed_value_counts.index)
                        
                        col_analysis["category_comparison"] = {
                            "seed_categories": len(seed_categories),
                            "synthetic_categories": len(synthetic_categories),
                            "new_categories": len(synthetic_categories - seed_categories),
                            "preserved_categories": len(synthetic_categories & seed_categories),
                            "jaccard_similarity": round(len(synthetic_categories & seed_categories) / len(synthetic_categories | seed_categories), 4) if len(synthetic_categories | seed_categories) > 0 else 0
                        }
                
                elif sdtype == 'datetime':
                    # Datetime analysis
                    try:
                        datetime_series = pd.to_datetime(synthetic_series, errors='coerce')
                        col_analysis.update({
                            "date_range": {
                                "min_date": str(datetime_series.min()) if not datetime_series.isna().all() else None,
                                "max_date": str(datetime_series.max()) if not datetime_series.isna().all() else None,
                                "date_span_days": (datetime_series.max() - datetime_series.min()).days if not datetime_series.isna().all() else None
                            },
                            "temporal_patterns": {
                                "year_distribution": datetime_series.dt.year.value_counts().head(10).to_dict() if not datetime_series.isna().all() else {},
                                "month_distribution": datetime_series.dt.month.value_counts().to_dict() if not datetime_series.isna().all() else {},
                                "day_of_week_distribution": datetime_series.dt.day_of_week.value_counts().to_dict() if not datetime_series.isna().all() else {}
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Datetime analysis failed for column {col}: {e}")
                
                table_analysis["column_distributions"][col] = col_analysis
            
            # Data quality metrics
            table_analysis["data_quality_metrics"] = {
                "completeness_score": round((1 - synthetic_df.isnull().sum().sum() / (len(synthetic_df) * len(synthetic_df.columns))) * 100, 2),
                "uniqueness_score": round(synthetic_df.nunique().sum() / (len(synthetic_df) * len(synthetic_df.columns)) * 100, 2),
                "consistency_score": round((1 - synthetic_df.duplicated().sum() / len(synthetic_df)) * 100, 2) if len(synthetic_df) > 0 else 100
            }
            
            analysis_report["table_analyses"][table_name] = table_analysis
        
        # Overall data quality assessment
        avg_completeness = np.mean([table["data_quality_metrics"]["completeness_score"] for table in analysis_report["table_analyses"].values()])
        avg_uniqueness = np.mean([table["data_quality_metrics"]["uniqueness_score"] for table in analysis_report["table_analyses"].values()])
        avg_consistency = np.mean([table["data_quality_metrics"]["consistency_score"] for table in analysis_report["table_analyses"].values()])
        
        analysis_report["data_quality_metrics"] = {
            "overall_completeness_score": round(avg_completeness, 2),
            "overall_uniqueness_score": round(avg_uniqueness, 2),
            "overall_consistency_score": round(avg_consistency, 2),
            "overall_quality_score": round((avg_completeness + avg_uniqueness + avg_consistency) / 3, 2)
        }
        
        logger.info("Data distribution analysis completed successfully")
        return analysis_report
        
    except Exception as e:
        logger.error(f"Error in data distribution analysis: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "generation_timestamp": datetime.now().isoformat()
        }

def collect_synthesis_metrics(start_time: float, end_time: float, synthetic_data: Dict[str, pd.DataFrame], 
                            metadata_dict: Dict[str, Any], synthesis_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect comprehensive synthesis performance metrics.
    """
    logger.info("Collecting synthesis performance metrics...")
    
    try:
        duration = end_time - start_time
        total_records = sum(len(df) for df in synthetic_data.values())
        total_memory = sum(df.memory_usage(deep=True).sum() for df in synthetic_data.values())
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metrics = {
            "performance_metrics": {
                "synthesis_time_seconds": round(duration, 2),
                "synthesis_time_formatted": f"{int(duration // 60)}m {int(duration % 60)}s",
                "records_per_second": round(total_records / duration, 2) if duration > 0 else 0,
                "memory_usage_mb": round(total_memory / (1024 * 1024), 2),
                "throughput_mb_per_second": round((total_memory / (1024 * 1024)) / duration, 2) if duration > 0 else 0
            },
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": round(memory_info.total / (1024 ** 3), 2),
                "memory_used_gb": round(memory_info.used / (1024 ** 3), 2),
                "memory_available_gb": round(memory_info.available / (1024 ** 3), 2),
                "memory_usage_percent": memory_info.percent
            },
            "synthesis_configuration": {
                "batch_size": synthesis_params.get("batch_size", "N/A"),
                "use_fast_synthesizer": synthesis_params.get("use_fast_synthesizer", "N/A"),
                "synthesizer_type": "Optimized SDV",
                "total_tables": len(synthetic_data),
                "total_records_generated": total_records
            },
            "efficiency_metrics": {
                "records_per_table": round(total_records / len(synthetic_data), 2) if len(synthetic_data) > 0 else 0,
                "memory_per_record_bytes": round(total_memory / total_records, 2) if total_records > 0 else 0,
                "generation_efficiency_score": round((total_records / duration) / (total_memory / (1024 * 1024)), 2) if duration > 0 and total_memory > 0 else 0
            }
        }
        
        logger.info("Synthesis metrics collection completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Error collecting synthesis metrics: {e}")
        return {
            "error": f"Metrics collection failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
