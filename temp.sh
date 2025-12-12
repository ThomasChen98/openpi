uv run examples/h1_control_client/embodied_reward_labeling.py \
    --data_dir examples/h1_control_client/h1_data_auto/lift_lid_reward_dec11/epoch_0/raw \
    --task_instruction "Lift the lid on top of the bowl with both hands, move it steadily away from the bowl, and place it flat on the table.
    The robot must use its two hands to do carefully put the lid on the table.
    Very important: The task fails if the lid slips, flipped, drops, or is knocked off during lifting or placement.
    The task fails if the lid is not placed flat on the table (e.g., still connected to the bowl).
    The task fails if the hands of the robot are colliding with the bowl or the table.
    If the task fails, all the frames after the failure should be scored 0." \
    --advantage_threshold 0.3