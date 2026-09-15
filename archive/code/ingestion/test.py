import urllib.request, json
res = urllib.request.urlopen('https://archive.ics.uci.edu/api/datasets?skip=0&take=5')
print(json.loads(res.read())['data'][0])
