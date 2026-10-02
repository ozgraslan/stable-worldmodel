"""Registration must not load the optional simulator or sibling repository."""

import subprocess
import sys


def test_registration_is_lazy():
    subprocess.run(
        [
            sys.executable,
            '-c',
            """
import sys
import importlib.abc
class RejectSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mani_skill', 'sapien', 'ppo', 'ppo_rgb', 'collect_ppo_dataset'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectSimulator())
import stable_worldmodel
import gymnasium
assert gymnasium.spec('swm/ManiSkillPushT-v1').entry_point == 'stable_worldmodel.envs.maniskill.pusht:PushTSWMEnv'
""",
        ],
        check=True,
    )
