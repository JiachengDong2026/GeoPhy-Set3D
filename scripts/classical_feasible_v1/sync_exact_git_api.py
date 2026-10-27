"""Preserve exact Git objects and fast-forward the authorized experiment ref via TLS API."""
import argparse,base64,concurrent.futures,datetime,hashlib,json,os,re,socket,subprocess,time
import urllib.request,urllib.error
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/scratch/djc/GeoPhy-Set3D/classical_feasible_v1')
REPOSITORY='JiachengDong2026/GeoPhy-Set3D'
BRANCH='experiment/classical_feasible_formal_v1'
original_resolver=socket.getaddrinfo


def resolve(host,port,*args,**kwargs):
    # DNS routing only; URL/SNI/certificate verification remain api.github.com.
    return original_resolver('20.205.243.168' if host=='api.github.com' else host,port,*args,**kwargs)


def git(*args,input=None):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0');env.pop('GH_DEBUG',None)
    result=subprocess.run(['git',*args],cwd=PROJECT,input=input,capture_output=True,env=env,timeout=30)
    if result.returncode:raise RuntimeError('Local Git command failed: '+args[0])
    return result.stdout


def credential():
    fields={}
    for line in git('credential','fill',input=b'protocol=https\nhost=github.com\n\n').decode().splitlines():
        if '=' in line:
            key,value=line.split('=',1);fields[key]=value
    assert fields.get('password'),'No noninteractive Git credential available'
    return fields['password']


def api(method,path,token,payload=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
    request=urllib.request.Request('https://api.github.com/repos/'+REPOSITORY+path,data=data,method=method,
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28','User-Agent':'GeoPhy-Set3D-exact-object-sync','Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(request,timeout=25) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in [429,500,502,503,504] or attempt==2:raise RuntimeError(f'GitHub API {method} {path}: HTTP {exc.code}') from None
        except (urllib.error.URLError,TimeoutError,OSError):
            if attempt==2:raise RuntimeError(f'GitHub API connection failed: {method} {path}') from None
        time.sleep(1+attempt)


def identity(raw):
    match=re.fullmatch(r'(.*) <([^>]*)> (\d+) ([+-]\d{4})',raw)
    assert match and match[4]=='+0000','Reject timezone rewriting rather than changing commit identities'
    date=datetime.datetime.fromtimestamp(int(match[3]),datetime.timezone.utc).isoformat(timespec='seconds')
    return dict(name=match[1],email=match[2],date=date)


def commit_payload(sha):
    raw=git('cat-file','commit',sha);head,_,message=raw.partition(b'\n\n');fields={};parents=[]
    for line in head.decode().splitlines():
        key,value=line.split(' ',1)
        assert key in ['tree','parent','author','committer'],'Unsupported commit header; do not rewrite'
        if key=='parent':parents.append(value)
        else:fields[key]=value
    assert hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==sha
    return dict(tree=fields['tree'],parents=parents,author=identity(fields['author']),
        committer=identity(fields['committer']),message=message.decode())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    socket.getaddrinfo=resolve;token=credential();head=git('rev-parse','HEAD').decode().strip()
    ref='/git/ref/heads/'+BRANCH;old=api('GET',ref,token)['object']['sha']
    if head==old:print('REMOTE_ALREADY_EXACT',head,flush=True);return
    git('merge-base','--is-ancestor',old,head)
    lines=git('rev-list','--objects',old+'..'+head).decode().splitlines()
    identifiers=[line.split(' ',1)[0] for line in lines]
    kinds=git('cat-file','--batch-check=%(objectname) %(objecttype)',input=('\n'.join(identifiers)+'\n').encode()).decode().splitlines()
    objects=dict(line.split() for line in kinds);commits=git('rev-list','--topo-order','--reverse',old+'..'+head).decode().splitlines()
    payloads={sha:commit_payload(sha) for sha in commits}
    plan=dict(remote_before=old,local_head=head,branch=BRANCH,objects=len(objects),
        blobs=sum(kind=='blob' for kind in objects.values()),trees=sum(kind=='tree' for kind in objects.values()),
        commits=commits,no_force=True,all_returned_object_shas_must_match=True,TLS_verification=True)
    (ROOT/'git_api_exact_sync_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('EXACT_OBJECT_PLAN',json.dumps(plan),flush=True)
    if not args.apply:return
    def blob(sha):
        value=api('POST','/git/blobs',token,dict(content=base64.b64encode(git('cat-file','blob',sha)).decode(),encoding='base64'))
        assert value['sha']==sha,('blob identity mismatch',sha)
        return sha
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for sha in pool.map(blob,[sha for sha,kind in objects.items() if kind=='blob']):print('BLOB_OK',sha,flush=True)
    completed=set()
    def tree(sha):
        if sha in completed or objects.get(sha)!='tree':return
        entries=[]
        for raw in git('ls-tree','-z',sha).split(b'\0'):
            if not raw:continue
            metadata,name=raw.split(b'\t',1);mode,kind,child=metadata.decode().split()
            if kind=='tree':tree(child)
            entries.append(dict(path=name.decode(),mode=mode,type=kind,sha=child))
        value=api('POST','/git/trees',token,dict(tree=entries));assert value['sha']==sha,('tree identity mismatch',sha)
        completed.add(sha);print('TREE_OK',sha,flush=True)
    for sha in commits:
        tree(payloads[sha]['tree'])
        value=api('POST','/git/commits',token,payloads[sha]);assert value['sha']==sha,('commit identity mismatch; ref untouched',sha)
        print('COMMIT_OK',sha,flush=True)
    current=api('GET',ref,token)['object']['sha'];assert current==old,'Concurrent ref change; do not overwrite'
    value=api('PATCH','/git/refs/heads/'+BRANCH,token,dict(sha=head,force=False));assert value['object']['sha']==head
    assert api('GET',ref,token)['object']['sha']==head
    tracking='refs/remotes/origin/'+BRANCH
    prior=git('rev-parse',tracking).decode().strip();git('update-ref',tracking,head,prior)
    (ROOT/'git_api_exact_sync_result.json').write_text(json.dumps(dict(passed=True,**plan,remote_after=head),indent=2)+'\n')
    print('EXACT_FAST_FORWARD_VERIFIED',head,flush=True)


if __name__=='__main__':main()
