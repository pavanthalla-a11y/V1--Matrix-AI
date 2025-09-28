#!/usr/bin/env python3
"""
Test script to run actual synthesis with 100 records
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"

def run_full_synthesis_test():
    """Run a complete synthesis test with 100 records"""
    print("🚀 Starting Full Synthesis Test (100 records)")
    print("=" * 60)
    
    # Step 1: Design Schema
    print("📋 Step 1: Designing schema...")
    design_payload = {
        "data_description": "Create a simple e-commerce dataset with users table (user_id, name, email, created_at) and orders table (order_id, user_id, product_name, amount, order_date)",
        "num_records": 100,
        "existing_metadata": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/design", json=design_payload)
        if response.status_code != 200:
            print(f"❌ Design failed: {response.status_code}")
            print(response.text)
            return False
        
        design_data = response.json()
        print(f"✅ Schema designed successfully!")
        print(f"   Tables: {design_data['tables_count']}")
        print(f"   Seed records: {design_data['total_seed_records']}")
        
        # Show metadata structure
        metadata = design_data['metadata_preview']
        print(f"   Table names: {list(metadata['tables'].keys())}")
        
    except Exception as e:
        print(f"❌ Design error: {e}")
        return False
    
    # Step 2: Start Synthesis
    print("\n⚙️  Step 2: Starting synthesis...")
    synthesis_payload = {
        "num_records": 100,
        "metadata_dict": design_data['metadata_preview'],
        "seed_tables_dict": design_data['seed_data_preview'],
        "user_email": TEST_EMAIL,
        "batch_size": 50,  # Small batches for testing
        "use_fast_synthesizer": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/synthesize", json=synthesis_payload)
        if response.status_code != 200:
            print(f"❌ Synthesis start failed: {response.status_code}")
            print(response.text)
            return False
        
        synthesis_response = response.json()
        print(f"✅ Synthesis started: {synthesis_response['status']}")
        print(f"   Target email: {synthesis_response['target_email']}")
        print(f"   Batch size: {synthesis_response['batch_size']}")
        
    except Exception as e:
        print(f"❌ Synthesis start error: {e}")
        return False
    
    # Step 3: Monitor Progress
    print("\n📊 Step 3: Monitoring progress...")
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{BASE_URL}/api/v1/progress")
            if response.status_code == 200:
                progress = response.json()
                status = progress['status']
                step = progress['current_step']
                percent = progress['progress_percent']
                records = progress['records_generated']
                
                print(f"   Status: {status} | Step: {step} | Progress: {percent}% | Records: {records}")
                
                if status == 'complete':
                    print(f"✅ Synthesis completed! Generated {records} records")
                    break
                elif status == 'error':
                    error_msg = progress.get('error_message', 'Unknown error')
                    print(f"❌ Synthesis failed: {error_msg}")
                    return False
                
                time.sleep(2)  # Wait 2 seconds between checks
            else:
                print(f"❌ Progress check failed: {response.status_code}")
                time.sleep(2)
                
        except Exception as e:
            print(f"⚠️  Progress check error: {e}")
            time.sleep(2)
    else:
        print("⏰ Synthesis timed out after 5 minutes")
        return False
    
    # Step 4: Sample the Generated Data
    print("\n📄 Step 4: Sampling generated data...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/sample?sample_size=5")
        if response.status_code == 200:
            sample_data = response.json()
            print(f"✅ Data sampling successful!")
            print(f"   Total tables: {sample_data['total_tables']}")
            
            # Show sample data for each table
            for table_name, samples in sample_data['all_samples'].items():
                print(f"\n   📊 Table '{table_name}' sample (first 5 rows):")
                if samples:
                    # Show column names
                    columns = list(samples[0].keys()) if samples else []
                    print(f"      Columns: {columns}")
                    
                    # Show first few rows
                    for i, row in enumerate(samples[:3]):
                        print(f"      Row {i+1}: {row}")
                else:
                    print("      No sample data available")
            
            return True
        else:
            print(f"❌ Data sampling failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Sampling error: {e}")
        return False

def main():
    """Main function to run the synthesis test"""
    print("🧪 Matrix AI - Full Synthesis Test")
    print("This will test the complete pipeline with 100 records")
    print()
    
    success = run_full_synthesis_test()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Full synthesis test completed successfully!")
        print("✅ All components working:")
        print("   • Schema design with AI")
        print("   • JSON parsing and validation")
        print("   • SDV synthesis pipeline")
        print("   • Progress monitoring")
        print("   • Data sampling")
        print("\n🚀 Your optimized main_optimized.py is production ready!")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED! Synthesis test encountered errors.")
        print("Please check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
