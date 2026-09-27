"""Run with /opt/private on PYTHONPATH inside a Task01/02 simulator image."""
import threading

import numpy as np

from harness.backend import Backend


def main():
    backend = Backend(task='OpenFridge', seed=0, split='pretrain')
    assert backend.video is None  # RLEBENCH_MEDIA=true must not record development.
    owner = threading.get_ident()
    pairs = 0
    try:
        for _ in range(3):
            backend.reset()
            sim = backend.env.sim
            context = sim._render_context_offscreen
            original_render = sim.render
            def render(**kwargs):
                assert threading.get_ident() == owner
                assert kwargs['width'] == kwargs['height'] == 512
                return original_render(**kwargs)
            sim.render = render
            buffers = context.con
            dimensions = (buffers.offWidth, buffers.offHeight)
            for _ in range(6):
                backend.frames.clear()
                for camera in backend.cameras:
                    rgb = backend.render_rgb(camera)
                    assert camera not in backend.frames
                    rgb_depth, depth = backend.render(camera)
                    np.testing.assert_array_equal(rgb, rgb_depth)
                    assert depth.shape == (512, 512) and np.isfinite(depth).all()
                    assert sim._render_context_offscreen is context and context.con is buffers
                    assert (buffers.offWidth, buffers.offHeight) == dimensions
                    pairs += 1
                action = [0.] * backend.info()['action_dim']
                action[6] = action[-1] = -1
                backend.act(action=action)
            sim.render = original_render
    finally:
        backend.close()
    print(f'{pairs} RGB/RGB-D pairs match; fixed render size, thread and framebuffer; development recording disabled')


if __name__ == '__main__':
    main()
