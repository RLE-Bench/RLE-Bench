"""Transfer-task stage evidence never enters a public observation."""
from .backend import Backend
from .compat import _fix_dump_leftovers_predicate
from . import stages


class Adapter(Backend):
    def create(self):
        _fix_dump_leftovers_predicate()
        return super().create()

    def after_reset(self):
        self.baseline = self.best = stages.read_stages(self.descriptor["task"], self.env)

    def evidence(self):
        success = bool(self.env._check_success())
        self.best = stages.merge_best(self.best, stages.read_stages(self.descriptor["task"], self.env))
        return dict(success=success, baseline=self.baseline, best=self.best,
                    score=stages.score_trial(self.baseline, self.best, success=success))
