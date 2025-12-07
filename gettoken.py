import http.client
import json

conn = http.client.HTTPSConnection("open-api.123pan.com")
payload = json.dumps({
    "clientID": "083388dfdba942ba98055be4a33c73ef",
    "clientSecret": "47ebe4edd84d4843964c95f723cb15bf"
})
headers = {
    'Platform': 'open_platform',
    'Content-Type': 'application/json'
}
conn.request("POST", "/api/v1/access_token", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))