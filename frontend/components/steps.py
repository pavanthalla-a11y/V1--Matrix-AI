import streamlit as st
import pandas as pd
import re
from ..api import call_api, FASTAPI_URL

def show_step1():
    st.header("STEP 1: DESIGN SCHEMA")
    st.caption("Use natural language to describe your required tables and relationships.")

    with st.container():
        col_desc, col_num = st.columns([3, 1])

        with col_desc:
            st.session_state.data_description = st.text_area(
                "Schema Description:",
                st.session_state.data_description,
                height=150,
                key="main_data_description_input"
            )
        
        with col_num:
            st.session_state.num_records = st.number_input(
                "Target Records:",
                min_value=1,
                max_value=1000000,
                value=st.session_state.num_records,
                step=100
            )
            
            estimated_time = max(1, st.session_state.num_records / 1000 * 0.5)
            st.markdown(f"**Est. Time:** ~{estimated_time:.1f} min")
            
            if st.button("GENERATE SCHEMA", use_container_width=True, type="primary"):
                if st.session_state.data_description:
                    existing_metadata = None
                    if st.session_state.ai_design_output:
                        existing_metadata = st.session_state.ai_design_output.get("metadata_dict")
                        st.toast("Refinement requested: Sending context to Gemini.")
                        
                    payload = {
                        "data_description": st.session_state.data_description,
                        "num_records": st.session_state.num_records,
                        "existing_metadata": existing_metadata
                    }
                    
                    with st.spinner("Calling Gemini to design schema and generate seed data..."):
                        response = call_api("POST", f"{FASTAPI_URL}/design", payload)

                    if response and response.get("status") == "review_required":
                        st.session_state.ai_design_output = {
                            "metadata_dict": response.get("metadata_preview"),
                            "seed_tables_dict": response.get("seed_data_preview"),
                        }
                        st.session_state.step = 2
                        st.session_state.synthesis_status = "Not Started"
                        st.success("SCHEMA DESIGN COMPLETE. Review the output below.")
                        st.rerun()

def show_step2():
    st.header("STEP 2: REVIEW SCHEMA")
    st.caption("Confirm the AI-generated metadata and seed data before synthesis.")
    
    ai_output = st.session_state.ai_design_output
    
    col_meta, col_seed = st.columns([1, 1])

    with col_meta:
        with st.container():
            st.subheader("METADATA STRUCTURE")
            st.json(ai_output["metadata_dict"], expanded=False)

    with col_seed:
        with st.container():
            st.subheader("SEED DATA SAMPLE")
            st.markdown("This sample trains the SDV model.")
            
            table_names = list(ai_output["seed_tables_dict"].keys())
            tabs = st.tabs(table_names)
            
            for i, table_name in enumerate(table_names):
                with tabs[i]:
                    df_seed = pd.DataFrame.from_records(ai_output["seed_tables_dict"][table_name])
                    st.dataframe(df_seed, use_container_width=True)

    st.subheader("LAUNCH SYNTHESIS")
    
    col_email, col_settings, col_start_btn = st.columns([2, 1, 1])

    with col_email:
        st.session_state.email = st.text_input(
            "Email for notifications:", 
            value=st.session_state.email, 
            key="user_email_input_2"
        )

    with col_settings:
        st.markdown("**Performance Settings:**")
        st.markdown(f"Batch Size: {st.session_state.batch_size:,}")
        st.markdown(f"Fast Mode: {'Yes' if st.session_state.use_fast_synthesizer else 'No'}")

    with col_start_btn:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("APPROVE & START", use_container_width=True, type="primary"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", st.session_state.email):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.step = 3
                st.rerun()

def show_step3():
    st.header("STEP 3: SYNTHESIS EXECUTION")
    
    if st.session_state.synthesis_status == "Not Started":
        if st.button(f"BEGIN SYNTHESIS FOR {st.session_state.num_records:,} RECORDS", use_container_width=True, type="secondary"):
            payload = {
                "num_records": st.session_state.num_records,
                "metadata_dict": st.session_state.ai_design_output["metadata_dict"],
                "seed_tables_dict": st.session_state.ai_design_output["seed_tables_dict"],
                "user_email": st.session_state.email,
                "batch_size": st.session_state.batch_size,
                "use_fast_synthesizer": st.session_state.use_fast_synthesizer
            }
            
            with st.spinner("Initiating optimized synthesis task..."):
                response = call_api("POST", f"{FASTAPI_URL}/synthesize", payload)

            if response and response.get("status") == "processing_started":
                st.session_state.synthesis_status = "Processing"
                st.success(f"OPTIMIZED SYNTHESIS STARTED! Notification will be sent to {st.session_state.email}")
                st.info("The process is running in the background. Proceed to Step 4 to check status.")
            
            st.rerun()

    elif st.session_state.synthesis_status == "Processing":
        st.warning(f"Synthesis is running in the background. You will receive an email at {st.session_state.email} when finished.")
        
        current_progress = get_progress_info()
        if current_progress["status"] == "processing":
            st.progress(current_progress["progress_percent"] / 100)
            st.markdown(f"**Current Step:** {current_progress['current_step']}")
            st.markdown(f"**Progress:** {current_progress['progress_percent']}%" )
            if current_progress["records_generated"] > 0:
                st.markdown(f"**Records Generated:** {current_progress['records_generated']:,}")

def show_step4():
    st.header("STEP 4: DATA FINALIZATION")

    if st.button("CHECK IF DATA IS READY", use_container_width=True, type="secondary"):
        ready_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
        
        if ready_response and ready_response.get("status") == "success":
            st.session_state.synthesis_status = "Complete"
            st.session_state.step = 4
            st.success("DATA IS READY! Review samples and store the data.")
            st.rerun()
        elif st.session_state.synthesis_status == "Processing":
            st.info("Data is still generating in the background. Please wait for the email notification.")
        else:
            st.error("Data is not yet ready or the background process failed. Check the server logs.")

    if st.session_state.synthesis_status == "Complete":
        st.subheader("COMPREHENSIVE ANALYTICS & REPORTS")
        
        # Get comprehensive reports from the new endpoint
        reports_response = call_api("GET", f"{FASTAPI_URL}/reports")
        
        if reports_response:
            # Display executive summary metrics
            synthesis_metrics = reports_response.get("synthesis_metrics", {})
            distribution_analysis = reports_response.get("distribution_analysis", {})
            
            if synthesis_metrics and "performance_metrics" in synthesis_metrics:
                perf_metrics = synthesis_metrics["performance_metrics"]
                config_metrics = synthesis_metrics.get("synthesis_configuration", {})
                sys_metrics = synthesis_metrics.get("system_metrics", {})
                
                st.markdown("#### 📊 PERFORMANCE METRICS")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Fix: total_records_generated is in synthesis_configuration, not performance_metrics
                    total_records = config_metrics.get('total_records_generated', 0)
                    st.metric(
                        "Total Records", 
                        f"{total_records:,}",
                        delta="Generated"
                    )
                
                with col2:
                    st.metric(
                        "Generation Time", 
                        perf_metrics.get('synthesis_time_formatted', 'N/A'),
                        delta=f"{perf_metrics.get('records_per_second', 0):.0f} rec/sec"
                    )
                
                with col3:
                    st.metric(
                        "Memory Usage", 
                        f"{perf_metrics.get('memory_usage_mb', 0):.1f} MB",
                        delta=f"{perf_metrics.get('throughput_mb_per_second', 0):.1f} MB/s"
                    )
                
                with col4:
                    efficiency_score = synthesis_metrics.get("efficiency_metrics", {}).get("generation_efficiency_score", 0)
                    st.metric(
                        "Efficiency Score", 
                        f"{efficiency_score:.2f}",
                        delta="Optimized"
                    )
            
            # Display data quality metrics
            if distribution_analysis and "data_quality_metrics" in distribution_analysis:
                quality_metrics = distribution_analysis["data_quality_metrics"]
                
                st.markdown("#### 🎯 DATA QUALITY ASSESSMENT")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    completeness = quality_metrics.get("overall_completeness_score", 0)
                    st.metric(
                        "Completeness", 
                        f"{completeness:.1f}%",
                        delta="Complete" if completeness > 95 else "Needs Review"
                    )
                
                with col2:
                    uniqueness = quality_metrics.get("overall_uniqueness_score", 0)
                    st.metric(
                        "Uniqueness", 
                        f"{uniqueness:.1f}%",
                        delta="Unique" if uniqueness > 80 else "Low Diversity"
                    )
                
                with col3:
                    consistency = quality_metrics.get("overall_consistency_score", 0)
                    st.metric(
                        "Consistency", 
                        f"{consistency:.1f}%",
                        delta="Consistent" if consistency > 90 else "Check Duplicates"
                    )
                
                with col4:
                    overall_quality = quality_metrics.get("overall_quality_score", 0)
                    quality_grade = "A" if overall_quality > 90 else "B" if overall_quality > 80 else "C"
                    st.metric(
                        "Overall Quality", 
                        f"{overall_quality:.1f}%",
                        delta=f"Grade {quality_grade}"
                    )
            
            # Analysis summary
            if distribution_analysis and "analysis_summary" in distribution_analysis:
                analysis_summary = distribution_analysis["analysis_summary"]
                
                st.markdown("#### 📈 SYNTHESIS OVERVIEW")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Tables Generated", 
                        analysis_summary.get("total_tables", 0),
                        delta="Multi-table"
                    )
                
                with col2:
                    amplification = analysis_summary.get("amplification_factor", 0)
                    st.metric(
                        "Data Amplification", 
                        f"{amplification:.1f}x",
                        delta=f"From {analysis_summary.get('total_seed_records', 0)} seed records"
                    )
                
                with col3:
                    memory_mb = analysis_summary.get("memory_usage_mb", 0)
                    st.metric(
                        "Dataset Size", 
                        f"{memory_mb:.1f} MB",
                        delta="In Memory"
                    )
        
        else:
            # Fallback to basic metrics if reports are not available
            sample_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
            
            if sample_response and sample_response.get("metadata"):
                metadata = sample_response["metadata"]
                
                st.markdown("#### 📊 BASIC METRICS")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Records", 
                        f"{metadata.get('total_records_generated', 0):,}",
                        delta="Generated"
                    )
                
                with col2:
                    generation_time = metadata.get('generation_time_seconds', 0)
                    st.metric(
                        "Generation Time", 
                        f"{generation_time:.1f}s",
                        delta=f"{metadata.get('total_records_generated', 0) / max(generation_time, 1):.0f} rec/sec"
                    )
                
                with col3:
                    st.metric(
                        "Tables Created", 
                        len(metadata.get('tables', {})),
                        delta="Tables"
                    )
                
                with col4:
                    avg_columns = sum(len(table.get('columns', [])) for table in metadata.get('tables', {}).values()) / max(len(metadata.get('tables', {})), 1)
                    st.metric(
                        "Avg Columns", 
                        f"{avg_columns:.0f}",
                        delta="Per Table"
                    )
        
        # Detailed Distribution Analysis Tabs
        if reports_response and distribution_analysis:
            st.subheader("📊 DETAILED DISTRIBUTION ANALYSIS")
            
            # Create tabs for different types of analysis
            analysis_tabs = st.tabs(["📈 Summary", "🔍 Column Analysis", "📋 Data Quality", "💾 Sample Data"])
            
            # Tab 1: Executive Summary
            with analysis_tabs[0]:
                st.markdown("#### Data Generation Summary")
                
                if "analysis_summary" in distribution_analysis:
                    summary = distribution_analysis["analysis_summary"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Generation Statistics:**")
                        st.markdown(f"- **Total Records Generated:** {summary.get('total_synthetic_records', 0):,}")
                        st.markdown(f"- **Original Seed Records:** {summary.get('total_seed_records', 0):,}")
                        st.markdown(f"- **Data Amplification:** {summary.get('amplification_factor', 0):.1f}x")
                        st.markdown(f"- **Tables Created:** {summary.get('total_tables', 0)}")
                        st.markdown(f"- **Memory Usage:** {summary.get('memory_usage_mb', 0):.1f} MB")
                    
                    with col2:
                        st.markdown("**📝 Data Description Used:**")
                        st.info(st.session_state.data_description)
                        
                        if synthesis_metrics and "synthesis_configuration" in synthesis_metrics:
                            config = synthesis_metrics["synthesis_configuration"]
                            st.markdown("**⚙️ Configuration:**")
                            st.markdown(f"- **Synthesizer:** {config.get('synthesizer_type', 'N/A')}")
                            st.markdown(f"- **Batch Size:** {config.get('batch_size', 'N/A'):,}")
                            st.markdown(f"- **Fast Mode:** {'Yes' if config.get('use_fast_synthesizer') else 'No'}")
            
            # Tab 2: Column Analysis
            with analysis_tabs[1]:
                st.markdown("#### Column-Level Distribution Analysis")
                
                if "table_analyses" in distribution_analysis:
                    table_analyses = distribution_analysis["table_analyses"]
                    
                    # Select table for detailed analysis
                    table_names = list(table_analyses.keys())
                    selected_table = st.selectbox("Select table for detailed analysis:", table_names)
                    
                    if selected_table and selected_table in table_analyses:
                        table_analysis = table_analyses[selected_table]
                        column_distributions = table_analysis.get("column_distributions", {})
                        
                        st.markdown(f"##### Analyzing Table: {selected_table}")
                        
                        # Basic table stats
                        basic_stats = table_analysis.get("basic_stats", {})
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Rows", f"{basic_stats.get('synthetic_rows', 0):,}")
                        with col2:
                            st.metric("Columns", basic_stats.get('synthetic_columns', 0))
                        with col3:
                            st.metric("Duplicates", basic_stats.get('duplicate_rows', 0))
                        with col4:
                            st.metric("Size (KB)", f"{basic_stats.get('memory_usage_kb', 0):.1f}")
                        
                        # Column details
                        st.markdown("##### Column Distribution Details")
                        
                        for col_name, col_analysis in column_distributions.items():
                            with st.expander(f"📊 Column: {col_name} ({col_analysis.get('sdtype', 'unknown')})"):
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    st.markdown("**Basic Statistics:**")
                                    st.markdown(f"- **Data Type:** {col_analysis.get('data_type', 'unknown')}")
                                    st.markdown(f"- **Unique Values:** {col_analysis.get('unique_values', 0):,}")
                                    st.markdown(f"- **Uniqueness Ratio:** {col_analysis.get('uniqueness_ratio', 0):.3f}")
                                    st.markdown(f"- **Null Values:** {col_analysis.get('null_count', 0)} ({col_analysis.get('null_percentage', 0):.1f}%)")
                                    
                                    # Type-specific statistics
                                    if 'mean' in col_analysis:  # Numerical column
                                        st.markdown("**Numerical Statistics:**")
                                        st.markdown(f"- **Mean:** {col_analysis.get('mean', 0):.3f}")
                                        st.markdown(f"- **Median:** {col_analysis.get('median', 0):.3f}")
                                        st.markdown(f"- **Std Dev:** {col_analysis.get('std', 0):.3f}")
                                        st.markdown(f"- **Min/Max:** {col_analysis.get('min', 0):.3f} / {col_analysis.get('max', 0):.3f}")
                                    
                                    elif 'top_categories' in col_analysis:  # Categorical column
                                        st.markdown("**Categorical Statistics:**")
                                        st.markdown(f"- **Categories:** {col_analysis.get('category_count', 0)}")
                                        if col_analysis.get('most_frequent'):
                                            most_freq = col_analysis['most_frequent']
                                            st.markdown(f"- **Most Frequent:** {most_freq.get('value', 'N/A')} ({most_freq.get('frequency', 0):.1%})")
                                
                                with col_right:
                                    # Distribution comparison with seed data
                                    if 'distribution_comparison' in col_analysis:
                                        comparison = col_analysis['distribution_comparison']
                                        st.markdown("**Distribution Comparison (vs Seed Data):**")
                                        st.markdown(f"- **Seed Mean:** {comparison.get('seed_mean', 0):.3f}")
                                        st.markdown(f"- **Synthetic Mean:** {comparison.get('synthetic_mean', 0):.3f}")
                                        st.markdown(f"- **Mean Difference:** {comparison.get('mean_difference', 0):.3f}")
                                        
                                        if 'statistical_tests' in col_analysis:
                                            ks_test = col_analysis['statistical_tests'].get('kolmogorov_smirnov', {})
                                            st.markdown(f"- **KS Test p-value:** {ks_test.get('p_value', 0):.4f}")
                                            st.markdown(f"- **Distribution Similarity:** {ks_test.get('interpretation', 'N/A')}")
                                    
                                    elif 'category_comparison' in col_analysis:
                                        comparison = col_analysis['category_comparison']
                                        st.markdown("**Category Comparison (vs Seed Data):**")
                                        st.markdown(f"- **Seed Categories:** {comparison.get('seed_categories', 0)}")
                                        st.markdown(f"- **Synthetic Categories:** {comparison.get('synthetic_categories', 0)}")
                                        st.markdown(f"- **Preserved:** {comparison.get('preserved_categories', 0)}")
                                        st.markdown(f"- **New Categories:** {comparison.get('new_categories', 0)}")
                                        st.markdown(f"- **Jaccard Similarity:** {comparison.get('jaccard_similarity', 0):.3f}")
                                    
                                    # Show top categories for categorical data
                                    if 'top_categories' in col_analysis:
                                        st.markdown("**Top Categories:**")
                                        top_cats = col_analysis['top_categories']
                                        for cat, count in list(top_cats.items())[:5]:
                                            st.markdown(f"- {cat}: {count}")
            
            # Tab 3: Data Quality Assessment
            with analysis_tabs[2]:
                st.markdown("#### Data Quality Assessment Report")
                
                if "data_quality_metrics" in distribution_analysis:
                    quality = distribution_analysis["data_quality_metrics"]
                    
                    # Overall quality visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("##### Overall Quality Scores")
                        
                        # Create quality bars
                        completeness = quality.get("overall_completeness_score", 0)
                        uniqueness = quality.get("overall_uniqueness_score", 0) 
                        consistency = quality.get("overall_consistency_score", 0)
                        overall = quality.get("overall_quality_score", 0)
                        
                        st.markdown(f"**Completeness:** {completeness:.1f}%")
                        st.progress(completeness / 100)
                        
                        st.markdown(f"**Uniqueness:** {uniqueness:.1f}%")
                        st.progress(uniqueness / 100)
                        
                        st.markdown(f"**Consistency:** {consistency:.1f}%") 
                        st.progress(consistency / 100)
                        
                        st.markdown(f"**Overall Quality:** {overall:.1f}%")
                        overall_color = "green" if overall > 90 else "orange" if overall > 75 else "red"
                        st.progress(overall / 100)
                    
                    with col2:
                        st.markdown("##### Quality Grade")
                        
                        grade = "A" if overall > 90 else "B" if overall > 80 else "C" if overall > 70 else "D"
                        grade_color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
                        
                        st.markdown(f"## {grade_color.get(grade, '⚪')} Grade {grade}")
                        st.markdown(f"**Score: {overall:.1f}%**")
                        
                        if overall > 90:
                            st.success("Excellent data quality!")
                        elif overall > 80:
                            st.warning("Good quality with minor issues")
                        elif overall > 70:
                            st.warning("Acceptable quality, review recommended")
                        else:
                            st.error("Poor quality, review required")
                    
                    # Table-level quality breakdown
                    if "table_analyses" in distribution_analysis:
                        st.markdown("##### Table-Level Quality Breakdown")
                        
                        quality_data = []
                        table_analyses = distribution_analysis["table_analyses"]
                        
                        for table_name, analysis in table_analyses.items():
                            table_quality = analysis.get("data_quality_metrics", {})
                            quality_data.append({
                                "Table": table_name,
                                "Completeness": f"{table_quality.get('completeness_score', 0):.1f}%",
                                "Uniqueness": f"{table_quality.get('uniqueness_score', 0):.1f}%", 
                                "Consistency": f"{table_quality.get('consistency_score', 0):.1f}%",
                                "Rows": analysis.get("basic_stats", {}).get("synthetic_rows", 0),
                                "Columns": analysis.get("basic_stats", {}).get("synthetic_columns", 0)
                            })
                        
                        if quality_data:
                            quality_df = pd.DataFrame(quality_data)
                            st.dataframe(quality_df, use_container_width=True)
            
            # Tab 4: Sample Data (existing functionality)
            with analysis_tabs[3]:
                st.markdown("#### Sample Data Preview")
                
                sample_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
                
                if sample_response and sample_response.get("all_samples"):
                    all_samples = sample_response["all_samples"]
                    sample_tabs = st.tabs(list(all_samples.keys()))
                    
                    for i, table_name in enumerate(all_samples.keys()):
                        with sample_tabs[i]:
                            st.markdown(f"##### Sample from {table_name}")
                            df_sample = pd.DataFrame(all_samples[table_name])
                            st.dataframe(df_sample, use_container_width=True)
                            
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.markdown(f"**Rows:** {len(df_sample):,}")
                                st.markdown(f"**Columns:** {len(df_sample.columns)}")
                            with col_stats2:
                                st.markdown(f"**Memory Usage:** {df_sample.memory_usage(deep=True).sum() / 1024:.1f} KB")
                                st.markdown(f"**Data Types:** {len(df_sample.dtypes.unique())} unique")
        
        else:
            # Fallback sample data preview if detailed analysis is not available
            st.subheader("SAMPLE DATA PREVIEW")
            
            sample_response = call_api("GET", f"{FASTAPI_URL}/sample", params={"sample_size": 20})
            
            if sample_response and sample_response.get("all_samples"):
                all_samples = sample_response["all_samples"]
                tabs = st.tabs(list(all_samples.keys()))
                
                for i, table_name in enumerate(all_samples.keys()):
                    with tabs[i]:
                        st.subheader(f"Sample from {table_name}")
                        df_sample = pd.DataFrame(all_samples[table_name])
                        st.dataframe(df_sample, use_container_width=True)
                        
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.markdown(f"**Rows:** {len(df_sample):,}")
                            st.markdown(f"**Columns:** {len(df_sample.columns)}")
                        with col_stats2:
                            st.markdown(f"**Memory Usage:** {df_sample.memory_usage(deep=True).sum() / 1024:.1f} KB")
                            st.markdown(f"**Data Types:** {len(df_sample.dtypes.unique())} unique")
        
        # Always show download section when data is ready
        st.subheader("📥 DATA DOWNLOAD")
        st.info("Download your generated synthetic data as a comprehensive ZIP package with all analysis reports.")

        col_download1, col_download2 = st.columns([3, 1])
        
        with col_download1:
            st.markdown("**Enhanced Download Package Includes:**")
            st.markdown("- 📊 All synthetic data tables as CSV files")
            st.markdown("- 📋 Metadata schema (JSON format)")
            st.markdown("- 🌱 Seed data used for training (JSON format)")
            st.markdown("- 📈 **NEW:** Synthesis performance metrics (JSON)")
            st.markdown("- 🔍 **NEW:** Data distribution analysis (JSON)")  
            st.markdown("- 📝 Enhanced generation summary with quality scores")
            st.markdown("- 🕒 Timestamped files for version control")
        
        with col_download2:
            if st.button("📥 DOWNLOAD ALL DATA", type="primary", use_container_width=True):
                with st.spinner("Preparing comprehensive download package..."):
                    try:
                        # Call the download endpoint
                        download_response = requests.get(f"{FASTAPI_URL}/download", timeout=300)
                        download_response.raise_for_status()
                        
                        # Generate filename with timestamp
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"matrix_ai_synthetic_data_{timestamp}.zip"
                        
                        # Create download button
                        st.download_button(
                            label="📥 CLICK TO DOWNLOAD ZIP FILE",
                            data=download_response.content,
                            file_name=filename,
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        st.balloons()
                        st.success("✅ Download package ready! Click the button above to save your comprehensive data package with analytics.")
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"Download failed: {e}")
                        st.info("Ensure your FastAPI server is running and data synthesis is complete.")
                    except Exception as e:
                        st.error(f"Unexpected error during download: {e}")
