"""Speed-run environment and the level-specific observation boundary."""
from .backend import Backend
from .privileged import privileged_state


class Adapter(Backend):
    def shown(self):
        public = super().shown()
        if self.descriptor.get("level") == "L3":
            public.update(privileged_state(self.raw, self.env))
        return public
