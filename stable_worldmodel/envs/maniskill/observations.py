"""Observation schema shared with ManiSkill PPO dataset collection."""


def state_dict(obs):
    # Preserve ManiSkill's state-only observation ordering, including privileged
    # task fields enabled by rgb+state_dict. Exclude sensor data and calibration.
    return {'agent': obs['agent'], 'extra': obs['extra']}


def flatten_policy_state(obs, device):
    from mani_skill.utils.common import flatten_state_dict

    return flatten_state_dict(state_dict(obs), use_torch=True, device=device)


def named_observation_leaves(obs):
    """Map Lance-safe column names to (original key path, batched value)."""
    leaves = {}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        else:
            column = 'obs_' + '_'.join(path).replace('.', '_')
            if column in leaves:
                raise ValueError(
                    f'Observation key paths collide in Lance column {column!r}'
                )
            leaves[column] = (path, value)

    visit(state_dict(obs), ())
    return leaves


def policy_observation(obs, device, policy_type):
    """Match RGB-only FlattenRGBDObservationWrapper without mutating stored observations."""
    if policy_type == 'state':
        return flatten_policy_state(obs, device)
    if policy_type != 'rgb':
        raise ValueError(f'Unknown policy type: {policy_type}')
    import torch

    return {
        'rgb': torch.cat(
            [
                camera['rgb'].to(device)
                for camera in obs['sensor_data'].values()
            ],
            dim=-1,
        )
    }
