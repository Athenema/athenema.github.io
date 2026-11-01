import json,gzip
d=json.load(open('tools/dots.json'))
h=open('tools/template.html').read().replace('__N__',str(d['n'])).replace('__GRID__',str(d['grid'])).replace('__PAL__',d['pal']).replace('__DATA__',d['data'])
open('mock/index.html','w').write(h)
print('html',len(h),'gz',len(gzip.compress(h.encode(),6)))
