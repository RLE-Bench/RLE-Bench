"""Infer the ballast quadrant from RGB-observed handle-lift deflection."""
import numpy as np


def solve_trial(sim):
    for position, gripper, steps in (
        ([0, 0, 1.05], -1, 60),
        ([0, 0, .86], -1, 60),
        ([0, 0, .86], 1, 40),
        ([0, 0, 1.06], 1, 80),
    ):
        obs = sim.move(position, [2**-.5, 2**-.5, 0, 0], gripper=gripper, steps=steps)
    rgb = obs['images']['top'].astype(float)
    yellow = ((rgb[:, :, 0] > 150) & (rgb[:, :, 1] > 100)
              & (rgb[:, :, 0]-rgb[:, :, 2] > 60)
              & (rgb[:, :, 1]-rgb[:, :, 2] > 50))
    rows, cols = np.where(yellow)
    if len(rows) < 200:
        raise RuntimeError('box is not visible')
    # The known handle-to-lid separation sets the approximate projection plane.
    lid_z = obs['eef_pos'][2] - .056
    scale = rgb.shape[0] / (2*np.tan(np.deg2rad(43/2)) * (1.71-lid_z))
    u, v = (cols.min()+cols.max())/2, (rows.min()+rows.max())/2
    cx, cy = (rgb.shape[1]-1)/2, (rgb.shape[0]-1)/2
    center = np.array([(u-cx)/scale+.03, -(v-cy)/scale])
    delta = center - np.array(obs['eef_pos'][:2])
    if np.min(np.abs(delta)) < .001:
        raise RuntimeError('lift deflection is inconclusive')
    answer = (('A' if delta[0] > 0 else 'B') if delta[1] < 0
              else ('C' if delta[0] > 0 else 'D'))
    return sim.submit(answer)


def solve(sim):
    while True:
        sim.observe()
        result = solve_trial(sim)
        print(result, flush=True)
        if result['done']:
            return result


if __name__ == '__main__':
    from harness.client import HiddenCOMClient
    solve(HiddenCOMClient())
