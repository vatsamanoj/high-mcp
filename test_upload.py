import requests

url = "http://localhost:8000/api/quotas/upload"
files = {'file': open('temp_quota.json', 'rb')}
response = requests.post(url, files=files)
print(response.json())
