# src/core/kinematics.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KinematicScreener:
    def __init__(self, fps=30):
        self.fps = fps

    def compute_energy(self, qpos_data):
        velocity = np.diff(qpos_data, axis=0, prepend=qpos_data[0:1])
        return np.sum(velocity ** 2, axis=1)

    def get_active_window(self, qpos_data, window_size=15, sensitivity=2.0):
        raw_energy = self.compute_energy(qpos_data)
        window = np.ones(window_size) / window_size
        smooth_energy = np.convolve(raw_energy, window, mode='same')
        
        noise_floor = np.mean(np.sort(smooth_energy)[:int(max(1, len(smooth_energy)*0.05))]) 
        noise_floor = max(noise_floor, 1e-6) 
        threshold = noise_floor * sensitivity
        
        active_indices = np.where(smooth_energy > threshold)[0]
        if len(active_indices) == 0:
            return 0, len(qpos_data) - 1

        start_frame = active_indices[0]
        end_frame = active_indices[-1]
        
        padding = int(0.5 * self.fps)
        return max(0, start_frame - padding), min(len(qpos_data) - 1, end_frame + padding)

    def select_key_frames_kmeans(self, qpos_data, active_start, active_end, num_frames=9):
        active_qpos = qpos_data[active_start:active_end+1]
        if len(active_qpos) < num_frames:
            return np.linspace(active_start, active_end, num_frames, dtype=int).tolist()
            
        scaler_pos = StandardScaler()
        qpos_scaled = scaler_pos.fit_transform(active_qpos)
        
        velocities = np.diff(active_qpos, axis=0, prepend=active_qpos[0:1])
        scaler_vel = StandardScaler()
        vel_scaled = scaler_vel.fit_transform(velocities)
        
        time_steps = np.arange(len(active_qpos)).reshape(-1, 1)
        scaler_time = StandardScaler()
        time_scaled = scaler_time.fit_transform(time_steps)
        
        W_pos, W_vel, W_time = 1.0, 2.0, 1.5  
        features = np.hstack([qpos_scaled * W_pos, vel_scaled * W_vel, time_scaled * W_time])
        
        kmeans = KMeans(n_clusters=num_frames, random_state=42, n_init=10)
        kmeans.fit(features)
        
        key_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            key_indices.append(np.argmin(distances) + active_start)
            
        key_indices = sorted(list(set(key_indices)))
        while len(key_indices) < num_frames:
            fallback = np.linspace(active_start, active_end, num_frames, dtype=int).tolist()
            key_indices = sorted(list(set(key_indices + fallback)))[:num_frames]
        return key_indices

def find_exact_transition_frame(qpos_window, global_start_idx, gripper_dim_indices, gripper_threshold=0.02):
    if len(qpos_window) < 3 or not gripper_dim_indices:
        return global_start_idx + len(qpos_window) // 2
    valid_gripper_dims = [d for d in gripper_dim_indices if -qpos_window.shape[1] <= d < qpos_window.shape[1]]
    if not valid_gripper_dims:
        return global_start_idx + len(qpos_window) // 2

    gripper_data = qpos_window[:, valid_gripper_dims]
    composite_velocity = np.linalg.norm(np.diff(gripper_data, axis=0), axis=1)
    
    if len(composite_velocity) > 0:
        max_change_idx = np.argmax(composite_velocity)
        if composite_velocity[max_change_idx] > gripper_threshold:
            return global_start_idx + max_change_idx

    all_dims = set(range(qpos_window.shape[1]))
    arm_dims = list(all_dims - set(valid_gripper_dims))
    if not arm_dims: return global_start_idx + len(qpos_window) // 2

    energy = np.sum(np.diff(qpos_window[:, arm_dims], axis=0) ** 2, axis=1)
    if len(energy) > 0:
        return global_start_idx + np.argmin(energy)
    return global_start_idx + len(qpos_window) // 2

def align_and_segment(vlm_json, indices_rel, qpos_data, dataset_start_offset, gripper_dim_indices, gripper_threshold):
    final_annotations = []
    for i, task in enumerate(vlm_json):
        img_start_idx = task["start_image"] - 1
        img_end_idx = task["end_image"] - 1
        
        rough_frame_start = indices_rel[img_start_idx]
        rough_frame_end = indices_rel[img_end_idx]
        window_qpos = qpos_data[rough_frame_start : rough_frame_end + 1]
        
        exact_end = find_exact_transition_frame(
            window_qpos, rough_frame_start, gripper_dim_indices, gripper_threshold
        )
        exact_start = rough_frame_start if i == 0 else final_annotations[-1]["exact_end_frame_relative"]
            
        final_annotations.append({
            "subtask_id": task["subtask_id"],
            "instruction": task["instruction"],
            "exact_start_frame_relative": int(exact_start),
            "exact_end_frame_relative": int(exact_end),
            "global_start_frame": int(dataset_start_offset + exact_start),
            "global_end_frame": int(dataset_start_offset + exact_end)
        })
    return final_annotations