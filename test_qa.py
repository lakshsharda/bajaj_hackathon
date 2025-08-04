#!/usr/bin/env python3
"""
Simple Q&A Test Script for HackRx System
"""

import requests
import json
import time

def test_hackrx_system():
    """Test the HackRx system with the competition format"""
    
    # Test configuration - Update this URL after deployment
    url = "https://your-app-name.azurewebsites.net/api/v1/hackrx/run"  # Replace with your actual Azure URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer b57bd62a8ac6975e085fe323f226a67b4cf72557d1b87eeb5c8daef5a1df1ecd"
    }
    
    # Test data (exact competition format)
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    print("🧪 TESTING HACKRX Q&A SYSTEM")
    print("=" * 50)
    print(f"📡 URL: {url}")
    print(f"📄 Document: Policy PDF")
    print(f"❓ Questions: {len(data['questions'])}")
    print()
    
    try:
        print("🔄 Sending request...")
        start_time = time.time()
        
        response = requests.post(url, json=data, headers=headers, timeout=300)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Request completed successfully")
            
            result = response.json()
            answers = result.get("answers", [])
            
            print("\n📋 ANSWERS:")
            print("-" * 50)
            
            for i, (question, answer) in enumerate(zip(data["questions"], answers), 1):
                print(f"{i}. Question: {question}")
                print(f"   Answer: {answer}")
                print()
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Request took too long")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Could not connect to server")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_hackrx_system() 