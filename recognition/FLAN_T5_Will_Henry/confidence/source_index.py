"""Metadata-only verified byte-offset index for prepared JSONL source manifests."""
import hashlib,json,os,pathlib,tempfile,sqlite3,contextlib

INDEX_DIR=pathlib.Path(os.environ.get('SOURCE_INDEX_DIR', '.source-index'))

def index_path(source, index_dir=INDEX_DIR):
    canonical=os.path.normcase(os.path.abspath(str(source)))
    return pathlib.Path(index_dir)/(hashlib.sha256(canonical.encode()).hexdigest()+'.json')

def database_path(source,index_dir=INDEX_DIR): return index_path(source,index_dir).with_suffix('.sqlite3')

def stamp(st): return {'size':st.st_size,'mtime_ns':st.st_mtime_ns}

def build(source, index_dir=INDEX_DIR):
    source=pathlib.Path(source); path=index_path(source,index_dir);path.parent.mkdir(parents=True,exist_ok=True)
    rows={}; digest=hashlib.sha256()
    with source.open('rb') as f:
        before=stamp(os.fstat(f.fileno()))
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            digest.update(line);row=json.loads(line.decode('utf-8-sig'));sid=row['id'];text=row['input']
            if not isinstance(sid,str) or not sid or sid in rows or not isinstance(text,str):raise ValueError('invalid or duplicate source ID')
            rows[sid]=[offset,hashlib.sha256(text.encode('utf-8')).hexdigest()]
        after=stamp(os.fstat(f.fileno()))
    if before!=after or stamp(source.stat())!=after:raise ValueError('source changed while indexing')
    payload={'version':1,'source_path':os.path.normcase(os.path.abspath(str(source))),'source_sha256':digest.hexdigest(),'source_stat':after,'entries':rows}
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.source-index-')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(payload,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    db=convert(source,index_dir)
    return {'index_path':str(db),'source_sha256':payload['source_sha256'],'count':len(rows),'source_stat':after}

def convert(source,index_dir=INDEX_DIR):
    source=pathlib.Path(source); payload=json.loads(index_path(source,index_dir).read_text(encoding='utf-8'))
    if payload.get('source_path')!=os.path.normcase(os.path.abspath(str(source))) or payload['source_stat']!=stamp(source.stat()):raise ValueError('stale source index')
    path=database_path(source,index_dir);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.source-db-');os.close(fd)
    try:
        with contextlib.closing(sqlite3.connect(tmp)) as db:
            with db:
                db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                db.execute('CREATE TABLE entries (id TEXT PRIMARY KEY,offset INTEGER NOT NULL,source_sha256 TEXT NOT NULL) WITHOUT ROWID')
                meta={k:payload[k] for k in ('version','source_path','source_sha256','source_stat')}
                db.executemany('INSERT INTO meta VALUES (?,?)',[(k,json.dumps(v)) for k,v in meta.items()])
                db.executemany('INSERT INTO entries VALUES (?,?,?)',((k,v[0],v[1]) for k,v in payload['entries'].items()))
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return path

def lookup(source, source_id, expected_source_sha256, index_dir=None):
    if index_dir is None:index_dir=INDEX_DIR
    source=pathlib.Path(source);uri=database_path(source,index_dir).resolve().as_uri()+'?mode=ro'
    with contextlib.closing(sqlite3.connect(uri,uri=True)) as db:
        idx={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
        item=db.execute('SELECT offset,source_sha256 FROM entries WHERE id=?',(source_id,)).fetchone()
    if idx.get('version')!=1 or idx.get('source_path')!=os.path.normcase(os.path.abspath(str(source))):raise ValueError('index source mismatch')
    with source.open('rb') as f:
        if stamp(os.fstat(f.fileno()))!=idx['source_stat'] or stamp(source.stat())!=idx['source_stat']:raise ValueError('stale source index')
        if not item or item[1]!=expected_source_sha256:raise ValueError('source ID or expected hash missing from index')
        f.seek(item[0]);row=json.loads(f.readline().decode('utf-8-sig'))
        if row.get('id')!=source_id or not isinstance(row.get('input'),str) or hashlib.sha256(row['input'].encode('utf-8')).hexdigest()!=expected_source_sha256:raise ValueError('indexed source row mismatch')
        if stamp(os.fstat(f.fileno()))!=idx['source_stat'] or stamp(source.stat())!=idx['source_stat']:raise ValueError('source changed during lookup')
    return row

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:raise SystemExit('Usage: py -3 source_index.py SOURCE_JSONL')
    print(json.dumps(build(sys.argv[1])))
