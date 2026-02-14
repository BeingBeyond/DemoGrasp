import torch

def reward_binary(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    has_hit_table,
    max_episode_length: float,
    table_heights,
    object_pos,
    palm_pos,
    fingertip_pos,
    num_fingers: int,
    object_init_states,
    **kwargs,
):
    info = {}
    object_delta_z = object_pos[:, 2] - object_init_states[:, 2]
    palm_object_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)
    horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1) #- object_init_states[:, 0:2], dim=-1)

    fingertips_object_dist = torch.zeros_like(object_delta_z)
    for i in range(fingertip_pos.shape[-2]):
        fingertips_object_dist += torch.norm(fingertip_pos[:, i, :] - object_pos, dim=-1)
    fingertips_object_dist = torch.where(fingertips_object_dist >= 3.0, 3.0, fingertips_object_dist)

    flag = torch.logical_or((fingertips_object_dist <= 0.12 * num_fingers), (palm_object_dist <= 0.15))

    # after hand approach object, lift_object
    lift_object = torch.zeros_like(object_delta_z)
    lift_object = torch.where(flag, object_delta_z, lift_object)

    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    progress_buf = torch.where(resets > 0, torch.zeros_like(progress_buf), progress_buf)
    
    successes = torch.where(
        object_delta_z > 0.1,
        torch.where(
            flag,
            torch.ones_like(successes),
            torch.zeros_like(successes),
        ),
        torch.zeros_like(successes),
    )
    current_successes = torch.where(resets>0, successes, current_successes)
    reward = current_successes.to(torch.float32) # binary success reward

    # check robot-table collision
    min_keypoint_z = torch.min(fingertip_pos[:, :, 2], dim=-1).values
    min_keypoint_z = torch.min(min_keypoint_z, palm_pos[:, 2])
    has_hit_table = torch.where(
        min_keypoint_z < table_heights,
        torch.ones_like(has_hit_table, dtype=torch.bool),
        has_hit_table
    )

    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["lift_object"] = lift_object
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = flag

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        has_hit_table,
        info,
    )


REWARD_DICT = {
    "binary": reward_binary
}