import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  "url": "https://i.imgur.com/5bGzZi7.jpg"
})
headers = {
  'X-API-KEY': '014af0003721206abe0b8fefabe154a6a5a38add',
  'Content-Type': 'application/json'
}
conn.request("POST", "/lens", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))