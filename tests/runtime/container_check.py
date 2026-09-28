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
    p.add_argument("--media", choices=("true", "false"), default="false")
    p.add_argument("--observe-plan", action="store_true", help="render every task02 trial instead of injecting faults")
    args = p.parse_args()
    name = "rlebench-redesign-"+uuid.uuid4().hex[:10]
    suffix = '-'+args.level.lower() if args.family == 'task01' else '-'+args.variant if args.variant else ''
    cmd = ['docker','run','-d','--name',name,'--label','rlebench.redesign=true',
           '--gpus','device='+args.gpu,'-e','RLEBENCH_TASK='+args.task,
           '-e','RLEBENCH_GROUP='+args.group,'-e','RLEBENCH_LEVEL='+args.level,
           '-e','RLEBENCH_MEDIA='+args.media]
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

    def check_media(count):
        index = json.loads(command(['cat', '/logs/verifier/media/index.json']))
        assert index['enabled'] == (args.media == 'true'), index
        expected = [f'trial-{i+1:02d}.mp4' for i in range(count)] if args.media == 'true' else []
        assert index['files'] == expected and index['skipped'] == [], index
        if expected:
            command(['python', '-'], code="""import os, pwd, subprocess
from pathlib import Path
import imageio_ffmpeg
root = Path('/logs/verifier')
owner = root.stat().st_uid or pwd.getpwnam('agent').pw_uid
for path in (root / 'media').glob('*'):
    assert path.stat().st_uid == owner, path
    if path.suffix != '.mp4': continue
    reader = imageio_ffmpeg.read_frames(str(path))
    info = next(reader)
    assert info['size'] == (768, 256) and info['fps'] == 10, info
    assert len(list(reader)) > 0
""")
        report['checks'].append(f'media={args.media}: {len(expected)} completed clips, playable and owned by output user')

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
        if args.variant == 'pocket':
            actor += '''    before=s.status()['steps_used']
    try: s._request('recover_drop')
    except RemoteError as e: assert e.kind=='bad_request'
    else: raise AssertionError('recovery accepted before drop')
    assert s.status()['steps_used']==before and not s.status()['episode_over']
    assert s.observe()['live']
    assert s.step(a,ObsSpec(cameras=()))['steps']==1
'''
        if args.oracle:
            actor = args.oracle.read_text()
        elif args.family == 'task01':
            actor += '''    remaining = s.status()['seconds_remaining']
    status = s.end_development()
    assert status['phase'] == 'evaluation'
    assert status['seconds_remaining'] <= remaining
    while s.status()['phase'] != 'finished':
        s.task_info()
        image = s.observe(ObsSpec(width=512,depth=True))
        assert image['live'] and image['resolution'] == [512,512], image
        s.step(a,ObsSpec(cameras=()))
        s.finish_trial()
        s.next_trial()
'''
        report['actor'] = agent(actor)
        report['checks'].append('real reset/step/render and lock')
        if args.family == 'task01':
            command(['mkdir','-p','/logs/verifier'])
            command(['chmod','777','/logs/verifier'])
            command(['/opt/control.sh','develop'])
            command(['/opt/verify.sh','develop'])
            reward = json.loads(command(['cat','/logs/verifier/reward.json']))
            assert reward['trials_recorded'] == reward['trials_total'], reward
            assert reward['infrastructure_failures'] == 0, reward
            check_media(reward['trials_recorded'])
            command(['python','/opt/check_isolation.py'],user='agent')
            report['reward'] = reward
            report['checks'].append('one client develops and evaluates all trials')
        if args.family == 'task03':
            command(['mkdir', '-p', '/logs/verifier'])
            command(['chmod', '777', '/logs/verifier'])
            command(['/opt/control.sh', 'attempt'])
            command(['/opt/verify.sh', 'attempt'])
            reward = json.loads(command(['cat', '/logs/verifier/reward.json']))
            assert reward['infrastructure_failures'] == 0, reward
            if args.variant == 'hidden-com':
                expected = {'correct', 'submitted', 'control_steps'}
                expected.update(f'trial_{i}_{key}' for i in range(1, 4)
                                for key in ('reward', 'correct', 'submitted', 'control_steps'))
            else:
                expected = {'quality', 'task_score', 'component_outcome', 'component_efficiency',
                            'interaction_steps', 'interaction_budget'}
                if args.variant == 'pocket':
                    expected.update(('optimal_qtm', 'actual_qtm', 'unclassified_transitions', 'recoveries'))
            assert expected <= reward.keys(), reward
            command(['python', '/opt/check_isolation.py'], user='agent')
            report['reward'] = reward
            report['checks'].append('private task metrics exported by verifier; UID isolation preserved')
        if args.family == 'task02':
            agent("from pathlib import Path\np=Path('/workspace/agent_harness/MANUAL.md');p.write_text('frozen development')\n")
            command(['/opt/control.sh','develop'])
            command(['mkdir','-p','/logs/verifier'])
            command(['chmod','777','/logs/verifier'])
            command(['/opt/verify.sh','develop'])
            check_media(0)
            if args.observe_plan:
                observations = []
                count = int(agent("from harness.client import SimClient\nwith SimClient() as s: print(s.status()['total_trials'])"))
                for i in range(count):
                    observations.append(agent('''from harness.client import SimClient, ObsSpec
from pathlib import Path
with SimClient() as s:
    s.task_info()
    assert not Path('/logs/verifier/diagnostics').exists()
    r=s.observe(ObsSpec(width=512,depth=True))
    assert r['live'] and r['ended'] is None, r
    assert sum(k.endswith('_image') for k in r['obs'])==3
    assert sum(k.endswith('_depth') for k in r['obs'])==3
    a=[0.]*s.task_info()['action_dim']
    s.step(a,ObsSpec(cameras=()))
    print(s.status()['trial_index'],r['resolution'])
'''))
                    command(['/opt/control.sh',f'eval_{i+1:02d}'])
                    command(['/opt/verify.sh',f'eval_{i+1:02d}'])
                    check_media(i+1)
                report['observations'] = observations
                report['reward'] = json.loads(command(['cat','/logs/verifier/reward.json']))
                assert report['reward']['infrastructure_failures'] == 0, report['reward']
                assert command(['test','-s','/logs/verifier/diagnostics/worker.log']) == ''
                command(['python','/opt/check_isolation.py'],user='agent')
                report['checks'].append('complete RGB/depth plan, handoffs and final diagnostics')
                report['ok'] = True
                return
            # The verifier must not retire the next trial while its worker is initializing.
            agent("from harness.client import SimClient\ns=SimClient();s.task_info();assert s.status()['trial_index']==0;assert not s.status()['episode_over'],s.status();s.disconnect()")
            agent("from pathlib import Path\nPath('/workspace/eval-only.txt').write_text('discard');Path('/workspace/agent_harness/MANUAL.md').write_text('changed')")
            kill = '''from pathlib import Path
import os, signal
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try: parts=p.read_bytes().split(b'\\0')
    except (FileNotFoundError,ProcessLookupError): continue
    if b'rlebench.runtime.worker' in parts: os.kill(int(p.parent.name),signal.SIGKILL)
'''
            command(['python','-'],code=kill)
            time.sleep(.5)
            agent("from harness.client import SimClient\ns=SimClient();assert s.status()['ended']=='sim_error',s.status();s.disconnect()")
            command(['/opt/control.sh','eval_01'])
            command(['/opt/verify.sh','eval_01'])
            index = json.loads(command(['cat','/logs/verifier/media/index.json']))
            assert index['files'] == [], index
            expected = [{'name': 'trial-01.mp4', 'reason': 'interrupted recording'}] if args.media == 'true' else []
            assert index['skipped'] == expected, index
            report['checks'].append('killed worker video skipped; active next trial excluded')
            agent("from harness.client import SimClient\nfrom pathlib import Path\nassert not Path('/workspace/eval-only.txt').exists();assert Path('/workspace/agent_harness/MANUAL.md').read_text()=='frozen development'\ns=SimClient();s.task_info();assert s.status()['trial_index']==1;assert not s.status()['episode_over'],s.status();s.disconnect()")
            report['checks'].append('worker SIGKILL isolates trial; next trial and frozen handoff survive')
            command(['python','-'],code=kill.replace('rlebench.runtime.worker','rlebench.runtime.server'))
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
