#!/usr/bin/env python3
"""
Stress Test Script for Matrix AI Optimized Performance
Tests various dataset sizes and configurations to validate performance improvements.
"""

import requests
import time
import json
import pandas as pd
from typing import Dict, Any, List
import threading
import concurrent.futures
from datetime import datetime

# Configuration
FASTAPI_URL = "http://localhost:8000/api/v1"
TEST_EMAIL = "test@example.com"

class StressTest:
    def __init__(self):
        self.results = []
        self.errors = []
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def call_api(self, method: str, endpoint: str, payload: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API call with error handling"""
        try:
            url = f"{FASTAPI_URL}/{endpoint}"
            response = requests.request(method, url, json=payload, params=params, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.errors.append(f"API Error: {str(e)}")
            return None
    
    def wait_for_completion(self, max_wait_time: int = 1800) -> bool:
        """Wait for synthesis to complete with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress = self.call_api("GET", "progress")
            if progress:
                status = progress.get("status", "idle")
                if status == "complete":
                    return True
                elif status == "error":
                    self.errors.append(f"Synthesis failed: {progress.get('error_message', 'Unknown error')}")
                    return False
                elif status == "processing":
                    self.log(f"Progress: {progress.get('progress_percent', 0)}% - {progress.get('current_step', 'Unknown')}")
            
            time.sleep(5)  # Check every 5 seconds
        
        self.errors.append(f"Timeout after {max_wait_time} seconds")
        return False
    
    def test_dataset_size(self, num_records: int, batch_size: int = 1000, description: str = None) -> Dict[str, Any]:
        """Test a specific dataset size"""
        if description is None:
            description = "A simple e-commerce database with customers, orders, and products tables with relationships."
        
        self.log(f"Testing {num_records:,} records with batch size {batch_size}")
        
        # Step 1: Design schema
        design_payload = {
            "data_description": description,
            "num_records": num_records,
            "existing_metadata": {}
        }
        
        start_time = time.time()
        design_response = self.call_api("POST", "design", design_payload)
        
        if not design_response or design_response.get("status") != "review_required":
            self.errors.append(f"Design failed for {num_records} records")
            return None
        
        design_time = time.time() - start_time
        self.log(f"Schema design completed in {design_time:.2f} seconds")
        
        # Step 2: Start synthesis
        synthesis_payload = {
            "num_records": num_records,
            "metadata_dict": design_response["metadata_preview"],
            "seed_tables_dict": design_response["seed_data_preview"],
            "user_email": TEST_EMAIL,
            "batch_size": batch_size,
            "use_fast_synthesizer": True
        }
        
        synthesis_start = time.time()
        synthesis_response = self.call_api("POST", "synthesize", synthesis_payload)
        
        if not synthesis_response or synthesis_response.get("status") != "processing_started":
            self.errors.append(f"Synthesis start failed for {num_records} records")
            return None
        
        # Step 3: Wait for completion
        if not self.wait_for_completion():
            return None
        
        synthesis_time = time.time() - synthesis_start
        
        # Step 4: Get results
        sample_response = self.call_api("GET", "sample", params={"sample_size": 10})
        
        if not sample_response or sample_response.get("status") != "success":
            self.errors.append(f"Failed to get samples for {num_records} records")
            return None
        
        metadata = sample_response.get("metadata", {})
        total_generated = metadata.get("total_records_generated", 0)
        generation_time = metadata.get("generation_time_seconds", synthesis_time)
        
        result = {
            "target_records": num_records,
            "actual_records": total_generated,
            "batch_size": batch_size,
            "design_time": design_time,
            "synthesis_time": synthesis_time,
            "generation_time": generation_time,
            "records_per_second": total_generated / max(generation_time, 0.1),
            "tables_created": len(metadata.get("tables", {})),
            "success": True
        }
        
        self.log(f"✅ {num_records:,} records completed in {synthesis_time:.1f}s ({result['records_per_second']:.0f} rec/sec)")
        return result
    
    def run_performance_tests(self):
        """Run a series of performance tests"""
        self.log("🚀 Starting Matrix AI Performance Stress Tests")
        self.log("=" * 60)
        
        # Test configurations: (records, batch_size, description)
        test_configs = [
            (100, 100, "Small test - single table with customers"),
            (500, 500, "Medium test - two related tables: customers and orders"),
            (1000, 1000, "Standard test - e-commerce database with 3 tables"),
            (2000, 1000, "Large test - subscription database with 4 tables"),
            (5000, 2000, "Stress test - complex multi-table database"),
        ]
        
        for num_records, batch_size, description in test_configs:
            try:
                result = self.test_dataset_size(num_records, batch_size, description)
                if result:
                    self.results.append(result)
                else:
                    self.log(f"❌ Test failed for {num_records:,} records")
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                self.errors.append(f"Test exception for {num_records} records: {str(e)}")
                self.log(f"❌ Exception in test for {num_records:,} records: {str(e)}")
        
        self.print_results()
    
    def test_concurrent_requests(self):
        """Test concurrent API requests"""
        self.log("🔄 Testing concurrent API requests")
        
        def make_design_request(i):
            payload = {
                "data_description": f"Test database {i} with customers and orders",
                "num_records": 100,
                "existing_metadata": {}
            }
            start = time.time()
            response = self.call_api("POST", "design", payload)
            duration = time.time() - start
            return {"request_id": i, "duration": duration, "success": response is not None}
        
        # Test 5 concurrent design requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_design_request, i) for i in range(5)]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful = sum(1 for r in concurrent_results if r["success"])
        avg_duration = sum(r["duration"] for r in concurrent_results) / len(concurrent_results)
        
        self.log(f"Concurrent test: {successful}/5 requests successful, avg duration: {avg_duration:.2f}s")
    
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        self.log("💾 Testing memory usage patterns")
        
        # Test with different batch sizes for same dataset
        batch_sizes = [500, 1000, 2000]
        num_records = 3000
        
        for batch_size in batch_sizes:
            self.log(f"Testing {num_records:,} records with batch size {batch_size}")
            result = self.test_dataset_size(num_records, batch_size, 
                                          "Memory test - large dataset with multiple tables")
            if result:
                self.log(f"Batch size {batch_size}: {result['records_per_second']:.0f} rec/sec")
    
    def print_results(self):
        """Print comprehensive test results"""
        self.log("\n" + "=" * 60)
        self.log("📊 STRESS TEST RESULTS SUMMARY")
        self.log("=" * 60)
        
        if not self.results:
            self.log("❌ No successful tests completed")
            return
        
        # Performance summary
        total_records = sum(r["actual_records"] for r in self.results)
        total_time = sum(r["synthesis_time"] for r in self.results)
        avg_speed = sum(r["records_per_second"] for r in self.results) / len(self.results)
        
        self.log(f"Total Records Generated: {total_records:,}")
        self.log(f"Total Time: {total_time:.1f} seconds")
        self.log(f"Average Speed: {avg_speed:.0f} records/second")
        self.log("")
        
        # Detailed results table
        self.log("Detailed Results:")
        self.log("-" * 80)
        self.log(f"{'Records':>8} {'Batch':>6} {'Time(s)':>8} {'Speed':>10} {'Tables':>7} {'Status'}")
        self.log("-" * 80)
        
        for result in self.results:
            status = "✅ OK" if result["success"] else "❌ FAIL"
            self.log(f"{result['target_records']:>8,} {result['batch_size']:>6} "
                    f"{result['synthesis_time']:>8.1f} {result['records_per_second']:>10.0f} "
                    f"{result['tables_created']:>7} {status}")
        
        # Performance comparison with original
        self.log("\n" + "=" * 60)
        self.log("📈 PERFORMANCE COMPARISON")
        self.log("=" * 60)
        
        # Estimate original performance (11+ hours for 1000 records)
        original_speed = 1000 / (11 * 3600)  # records per second
        
        for result in self.results:
            if result["target_records"] == 1000:
                improvement = result["records_per_second"] / original_speed
                self.log(f"1000 records: {result['synthesis_time']:.1f}s vs 11+ hours")
                self.log(f"Performance improvement: {improvement:.0f}x faster")
                break
        
        # Error summary
        if self.errors:
            self.log("\n" + "=" * 60)
            self.log("⚠️  ERRORS ENCOUNTERED")
            self.log("=" * 60)
            for error in self.errors:
                self.log(f"❌ {error}")
        
        self.log("\n🎉 Stress test completed!")

def main():
    """Main stress test execution"""
    print("Matrix AI - Performance Stress Test")
    print("=" * 40)
    print("This script will test the optimized performance improvements.")
    print("Make sure the FastAPI server is running on localhost:8000")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{FASTAPI_URL.replace('/api/v1', '')}/docs", timeout=5)
        if response.status_code != 200:
            print("❌ FastAPI server not responding. Please start the server first:")
            print("   uvicorn main_optimized:app --reload --port 8000")
            return
    except:
        print("❌ Cannot connect to FastAPI server. Please start the server first:")
        print("   uvicorn main_optimized:app --reload --port 8000")
        return
    
    print("✅ FastAPI server is running")
    print()
    
    # Run stress tests
    stress_test = StressTest()
    
    try:
        # Main performance tests
        stress_test.run_performance_tests()
        
        # Additional tests
        print("\n" + "=" * 60)
        stress_test.test_concurrent_requests()
        
        print("\n" + "=" * 60)
        stress_test.test_memory_usage()
        
    except KeyboardInterrupt:
        print("\n⚠️  Stress test interrupted by user")
    except Exception as e:
        print(f"\n❌ Stress test failed with error: {str(e)}")
    
    print("\nStress test completed. Check the results above.")

if __name__ == "__main__":
    main()
