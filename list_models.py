
import requests
import os
import json

API_KEY = "AIzaSyBaVDMRLXtv3e4_SsCx7h9MN9jr0XAolqY"
URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

def list_models():
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"Found {len(models)} models.")
            print("-" * 40)
            gemma_models = []
            for m in models:
                name = m.get('name', '')
                display_name = m.get('displayName', '')
                if 'gemma' in name.lower() or 'gemma' in display_name.lower():
                    gemma_models.append(m)
                    print(f"Name: {name}")
                    print(f"Display Name: {display_name}")
                    print(f"Description: {m.get('description', 'N/A')}")
                    print("-" * 20)
            
            if not gemma_models:
                print("No Gemma models found.")
            else:
                print(f"Found {len(gemma_models)} Gemma models.")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_models()
