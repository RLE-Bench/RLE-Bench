"""Real GPU/UID smoke and fault injection. Creates and removes only its own container."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="task02")
    p.add_argument("--task", default="")
    p.add_argument("--group", default="06-setting-the-table")
    p.add_argument("--variant", default="")
    p.add_argument("--level", default="L1")
    p.add_argument("--assets", type=Path)
    p.add_argument("--gpu", default="2")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--oracle", type=Path)
    args = p.parse_args()
    name = "rlebench-redesign-"+uuid.uuid4().hex[:10]
    suffix = '-'+args.level.lower() if args.family == 'task01' else '-'+args.variant if args.variant else ''
    cmd = ['docker','run','-d','--name',name,'--label','rlebench.redesign=true',
           '--gpus','device='+args.gpu,'-e','RLEBENCH_TASK='+args.task,
           '-e','RLEBENCH_GROUP='+args.group,'-e','RLEBENCH_LEVEL='+args.level]
    if args.assets:
        cmd += ['-v',str(args.assets.resolve())+':/opt/src/robocasa/robocasa/models/assets:ro']
    cmd += [f'rlebench-{args.family}{suffix}-agent:dev']
    report = dict(family=args.family, task=args.task, group=args.group, checks=[])

    def command(arguments, user='root', code=None):
        result = subprocess.run(['docker','exec','-i','--user',user,name,*arguments],
                                input=code, text=True, capture_output=True, timeout=380)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        return result.stdout

    def agent(code):
        return command(['python','-'], user='agent', code=code)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        for _ in range(180):
            result = subprocess.run(['docker','exec',name,'/opt/healthcheck.sh'],capture_output=True)
            if result.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('broker never became healthy')
        command(['python','/opt/check_isolation.py'],user='agent')
        report['checks'].append('UID isolation')
        actor = '''from harness.client import SimClient, ObsSpec, RemoteError
with SimClient() as s:
    if s.status()['phase']=='development': s.reset()
    info=s.task_info()
    print(info,flush=True)
    assert not s.status()['episode_over'], s.status()
    small=s.observe(ObsSpec(width=160,height=96))
    big=s.observe(ObsSpec(width=512))
    assert small['resolution']==[160,96] and big['resolution']==[512,512]
    try:
        other=SimClient()
    except RemoteError as e: assert e.kind=='busy'
    else: raise AssertionError('second controller admitted')
    a=[0.]*info['action_dim']
    result=s.step(a,ObsSpec(cameras=()))
    assert result['steps']==1 and result['ended'] is None, result
'''
        if args.variant == 'hidden-com':
            actor = '''from harness.client import HiddenCOMClient
s=HiddenCOMClient()
o=s.observe()
assert set(o['images'])=={'workspace','closeup','top'}
o=s.step([0.]*6+[-1.],steps=2)
assert o['steps']==2, o
r=s.submit('A')
assert r['submitted'] and not r['done']
try: s.step([0.]*7)
except Exception: pass
else: raise AssertionError('stale trial accepted')
s.observe();s.disconnect()
'''
        if args.oracle:
            actor = args.oracle.read_text()
        report['actor'] = agent(actor)
        report['checks'].append('real reset/step/render and lock')
        if args.family == 'task02':
            agent("from pathlib import Path\np=Path('/workspace/agent_harness/MANUAL.md');p.write_text('frozen development')\n")
            command(['/opt/control.sh','develop'])
            command(['mkdir','-p','/logs/verifier'])
            command(['chmod','777','/logs/verifier'])
            command(['/opt/verify.sh','develop'])
            # The verifier must not retire the next trial while its worker is initializing.
            agent("from harness.client import SimClient\ns=SimClient();s.task_info();assert s.status()['trial_index']==0;assert not s.status()['episode_over'],s.status();s.disconnect()")
            agent("from pathlib import Path\nPath('/workspace/eval-only.txt').write_text('discard');Path('/workspace/agent_harness/MANUAL.md').write_text('changed')")
            kill = '''from pathlib import Path
import os, signal
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: parts=p.read_bytes().split(b'\\0')
    except (FileNotFoundError,ProcessLookupError): continue
    if b'harness.runtime.worker' in parts: os.kill(int(p.parent.name),signal.SIGKILL)
'''
            command(['python','-'],code=kill)
            time.sleep(.5)
            agent("from harness.client import SimClient\ns=SimClient();assert s.status()['ended']=='sim_error',s.status();s.disconnect()")
            command(['/opt/control.sh','eval_01'])
            command(['/opt/verify.sh','eval_01'])
            agent("from harness.client import SimClient\nfrom pathlib import Path\nassert not Path('/workspace/eval-only.txt').exists();assert Path('/workspace/agent_harness/MANUAL.md').read_text()=='frozen development'\ns=SimClient();s.task_info();assert s.status()['trial_index']==1;assert not s.status()['episode_over'],s.status();s.disconnect()")
            report['checks'].append('worker SIGKILL isolates trial; next trial and frozen handoff survive')
            command(['python','-'],code=kill.replace('harness.runtime.worker','harness.runtime.server'))
            time.sleep(2)
            agent("from harness.client import SimClient\ns=SimClient();assert s.status()['ended']=='sim_error',s.status();s.disconnect()")
            command(['/opt/control.sh','eval_02'])
            agent("from harness.client import SimClient\ns=SimClient();s.task_info();assert s.status()['trial_index']==2;assert not s.status()['episode_over'],s.status();s.disconnect()")
            report['checks'].append('broker SIGKILL recovers private ledger without action replay')
        report['ok'] = True
    except Exception as exc:
        report.update(ok=False,error=str(exc))
        try:
            report['service_log'] = command(['tail','-70','/var/lib/rlebench/service.log'])
            report['worker_log'] = command(['tail','-70','/var/lib/rlebench/worker.log'])
        except Exception:
            pass
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        subprocess.run(['docker','rm','-f',name],capture_output=True)


if __name__ == '__main__':
    main()
