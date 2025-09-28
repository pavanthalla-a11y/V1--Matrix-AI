#!/usr/bin/env python3
"""
Test script to verify JSON parsing fixes for multi-table scenarios
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_multi_table_parsing():
    """Test multi-table schema generation with improved JSON parsing"""
    print("🔧 Testing Multi-Table JSON Parsing Fixes")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Game Database (Players & Teams)",
            "description": "Create a game database with players table (player_id, name, email, registration_date) and teams table (team_id, team_name, captain_id, created_date)"
        },
        {
            "name": "Subscription Database",
            "description": "A multi-table subscription database with products, offers, subscriptions, and entitlements."
        },
        {
            "name": "E-commerce Database",
            "description": "Create an e-commerce database with customers, orders, products, and order_items tables with proper relationships"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        
        payload = {
            "data_description": test_case['description'],
            "num_records": 50,
            "existing_metadata": {}
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/design", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS: Schema generated")
                print(f"   Status: {data['status']}")
                print(f"   Tables: {data['tables_count']}")
                print(f"   Seed records: {data['total_seed_records']}")
                
                # Check metadata structure
                metadata = data['metadata_preview']
                table_names = list(metadata['tables'].keys())
                print(f"   Table names: {table_names}")
                
                # Check for relationships
                relationships = metadata.get('relationships', [])
                print(f"   Relationships: {len(relationships)}")
                
                # Verify datetime format strings are correct
                datetime_issues = []
                for table_name, table_info in metadata['tables'].items():
                    columns = table_info.get('columns', {})
                    for col_name, col_info in columns.items():
                        if col_info.get('sdtype') == 'datetime':
                            format_str = col_info.get('datetime_format', '')
                            if '"' in format_str or format_str.count('%') != format_str.count('%Y') + format_str.count('%m') + format_str.count('%d') + format_str.count('%H') + format_str.count('%M') + format_str.count('%S'):
                                datetime_issues.append(f"{table_name}.{col_name}: {format_str}")
                
                if datetime_issues:
                    print(f"   ⚠️  Datetime format issues found: {datetime_issues}")
                else:
                    print(f"   ✅ All datetime formats are correct")
                
                print(f"   Result: PASSED")
                
            else:
                print(f"❌ FAILED: {response.status_code}")
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', 'Unknown error')
                    print(f"   Error: {error_detail}")
                    
                    # Check if it's a JSON parsing error
                    if "JSON" in error_detail or "parse" in error_detail.lower():
                        print(f"   🔍 JSON parsing issue detected")
                        if "datetime_format" in error_detail:
                            print(f"   🔍 Datetime format corruption detected")
                    
                except:
                    print(f"   Raw error: {response.text}")
                
                print(f"   Result: FAILED")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            print(f"   Result: ERROR")
    
    print(f"\n" + "=" * 50)
    print(f"Multi-table JSON parsing test completed")

def test_download_endpoint():
    """Test if download endpoint is available"""
    print(f"\n📥 Testing Download Endpoint Availability")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/download")
        
        if response.status_code == 404:
            print(f"✅ Download endpoint correctly returns 404 when no data available")
        elif response.status_code == 400:
            print(f"✅ Download endpoint correctly returns 400 when synthesis in progress")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Download endpoint error: {e}")

def main():
    """Run JSON parsing tests"""
    print("🧪 Matrix AI - JSON Parsing & Multi-Table Test")
    print("Testing improved JSON validation and datetime format fixes")
    print()
    
    # Test multi-table scenarios
    test_multi_table_parsing()
    
    # Test download endpoint
    test_download_endpoint()
    
    print(f"\n🎯 Test completed!")
    print(f"If all tests passed, the JSON parsing fixes are working correctly.")

if __name__ == "__main__":
    main()
