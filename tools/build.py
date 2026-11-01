import json,gzip,sys,os
src=sys.argv[1] if len(sys.argv)>1 else 'tools/dots.json'
out=sys.argv[2] if len(sys.argv)>2 else 'mock'
os.makedirs(out,exist_ok=True)
d=json.load(open(src))
h=open('tools/template.html').read().replace('__N__',str(d['n'])).replace('__GRID__',str(d['grid'])).replace('__LO__',repr(round(d['lo'],5))).replace('__HI__',repr(round(d['hi'],5))).replace('__SIGMA__',json.dumps(d['scales'])).replace('__PAL__',d['pal']).replace('__DATA__',d['data'])
open(out+'/index.html','w').write(h)
print('html',len(h),'gz',len(gzip.compress(h.encode(),6)))
