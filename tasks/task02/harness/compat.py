"""Compatibility fixes for the pinned simulator commits."""
def _fix_render_cleanup() -> None:
    """Free GL resources in their owning context, preserving the caller's context."""
    try:
        from robosuite.utils.binding_utils import MjRenderContext
    except ImportError:
        return
    original = MjRenderContext.__del__
    if getattr(original, '_context_safe', False):
        return

    def cleanup(self):
        if not hasattr(self, 'con'):
            return
        import os
        import ctypes
        previous = None
        if os.environ.get('MUJOCO_GL') == 'egl':
            from OpenGL import EGL
            previous = (EGL.eglGetCurrentDisplay(), EGL.eglGetCurrentSurface(EGL.EGL_DRAW), EGL.eglGetCurrentSurface(EGL.EGL_READ), EGL.eglGetCurrentContext())
        self.gl_ctx.make_current()
        if previous is not None:
            previous_id = ctypes.cast(previous[3], ctypes.c_void_p).value
            owned_id = ctypes.cast(EGL.eglGetCurrentContext(), ctypes.c_void_p).value
        try:
            original(self)
        finally:
            if previous is not None and previous[0] and (previous_id != owned_id):
                EGL.eglMakeCurrent(*previous)
    cleanup._context_safe = True
    MjRenderContext.__del__ = cleanup

def _fix_dump_leftovers_predicate() -> None:
    """DumpLeftovers' shipped `_check_success` tests `leftover1` twice and never
    `leftover2`, so the bowl only had to shed one item. Require both, as the instruction
    reads. `stages.dump_leftovers` mirrors this corrected predicate. Idempotent; no-op
    off-simulator."""
    try:
        from robocasa.environments.kitchen.composite.washing_dishes.dump_leftovers import DumpLeftovers
        import robocasa.utils.object_utils as OU
    except ImportError:
        return
    if getattr(DumpLeftovers, '_rlebench_fixed', False):
        return

    def _check_success(self):
        leftovers_dumped = not any((OU.check_obj_in_receptacle(self, name, 'bowl') for name in ('leftover1', 'leftover2')))
        bowl_in_sink = OU.obj_inside_of(self, 'bowl', self.sink)
        gripper_far = OU.gripper_obj_far(self, obj_name='bowl')
        return leftovers_dumped and bowl_in_sink and gripper_far
    DumpLeftovers._check_success = _check_success
    DumpLeftovers._rlebench_fixed = True
