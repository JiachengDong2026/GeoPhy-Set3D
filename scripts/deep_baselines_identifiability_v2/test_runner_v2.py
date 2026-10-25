"""Record CPU engineering tests, no optimizer constructed or stepped."""
from ctx_v2 import *
import subprocess

def main():
    p=ROOT/'cpu_engineering_tests.log'
    with p.open('x') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_baseline_identifiability_v2.py'],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    freeze(ROOT/'engineering_tests.json',dict(at=now(),passed=result.returncode==0,returncode=result.returncode,
        log_sha256=sha(p),optimizer_updates=0,test_scope='5 engineering tests;full64 backward on narrow models,full configured GPU backward separately'))
    print(p.read_text(),flush=True)
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
