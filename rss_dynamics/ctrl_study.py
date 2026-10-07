"""Belt ratio N and encoder: closed-loop tracking (showcase 2x, 50 g, 8 s) and pHRI
(force estimation) in the MuJoCo 'real' robot.   python3 ctrl_study.py d35_core"""
import json, os, sys
import numpy as np
from robot_setup import setup, HERE
from closed_loop import tracking, phri

name = sys.argv[1] if len(sys.argv) > 1 else 'd35_core'
cases = [(3, True, 'N=3, encoder del motor'), (4, True, 'N=4, encoder del motor'),
         (6, True, 'N=6, encoder del motor'), (3, 16384, 'N=3 + encoder 14 bit en la manivela'),
         (4, 16384, 'N=4 + encoder 14 bit en la manivela')]
out = []
for N, enc, lab in cases:
    S = setup(name, payload=(0.05, 0.05), mode='real', motors=[], N=N)
    try:
        L = tracking(S, speed=2.0, feedforward=True, T=8.0, quantise=enc, fc=30)
        tr = dict(ep_max_mm=float(L['ep'].max() * 1e3), eR_max_deg=float(np.degrees(L['eR'].max())),
                  tau_max=float(np.abs(L['tau']).max()), fb_rms=float(np.sqrt(((L['tau'] - L['tau_ff']) ** 2).mean())))
    except Exception as e:
        tr = dict(error=str(e))
    S = setup(name, mode='real', motors=[], N=N)
    try:
        P = phri(S, T=10.0, quantise=enc)
        e = P['w_est'][:, :3] - P['f_true']
        ph = dict(force_err_max_N=float(np.abs(e).max()), force_err_rms_N=float(np.sqrt((e ** 2).mean())),
                  disp_max_mm=float(np.abs(P['dp']).max() * 1e3), tau_max=float(np.abs(P['tau']).max()))
    except Exception as e:
        ph = dict(error=str(e))
    out.append(dict(N=N, encoder=str(enc), label=lab, tracking=tr, phri=ph))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(os.path.join(HERE, 'results', f'ctrl_study_{name}.json'), 'w'), indent=1)
