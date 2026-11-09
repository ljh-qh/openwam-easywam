"""Adaptive TTY runner for ssh -tt worker34001 (never blocks forever).

Differences vs tty_send.py:
  * waits for the remote shell prompt (or 30s) before writing the script
  * hard timeout + kill, so a stuck proxy can never hang the caller
"""
import os
import re
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

secret = os.environ['EASYWAM_BASTION_PASSWORD']
script = sys.stdin.read()
env = os.environ.copy()
# Proxy connect string for the bastion (host/port/user), supplied by env.
proxy = os.environ.get('EASYWAM_BASTION_PROXY', '')
env['EASYWAM_BASTION_PASSWORD'] = secret + proxy
env['SSH_ASKPASS_REQUIRE'] = 'force'
env['DISPLAY'] = ':0'
d = tempfile.mkdtemp()
ap = Path(d) / 'ap'
ap.write_text('#!/bin/sh\nprintf "%s\\n" "$EASYWAM_BASTION_PASSWORD"\n')
ap.chmod(0o700)
env['SSH_ASKPASS'] = str(ap)

timeout_s = float(os.environ.get('TTY_TIMEOUT', '240'))
p = subprocess.Popen(
    ['ssh', '-tt', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=15',
     '-o', 'NumberOfPasswordPrompts=1', 'worker34001'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=env, bufsize=0)

out = b''
sent = False
start = time.time()
end = start + timeout_s
prompt_seen = False
while time.time() < end:
    r, _, _ = select.select([p.stdout], [], [], 0.5)
    if r:
        chunk = os.read(p.stdout.fileno(), 65536)
        if not chunk:
            break
        out += chunk
        if not prompt_seen and re.search(rb'@worker34001:.*[$#] ?', out):
            prompt_seen = True
            time.sleep(0.3)
            p.stdin.write(("stty -echo 2>/dev/null\n" + script + "\nexit\n").encode())
            p.stdin.flush()
            sent = True
    elif p.poll() is not None:
        break
    if not sent and time.time() - start > 30:
        try:
            p.stdin.write(("stty -echo 2>/dev/null\n" + script + "\nexit\n").encode())
            p.stdin.flush()
            sent = True
        except BrokenPipeError:
            break
p.kill()
try:
    p.wait(timeout=5)
except subprocess.TimeoutExpired:
    pass
text = out.decode('utf-8', 'replace')
text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
text = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', text).replace('\r', '')
lines = [ln for ln in text.splitlines()
         if not re.match(r'^mccuseradmin@worker34001:.*\$ ?$', ln)]
print('\n'.join(lines))
print(f'[tty_run elapsed={time.time()-start:.1f}s sent={sent}]', file=sys.stderr)
