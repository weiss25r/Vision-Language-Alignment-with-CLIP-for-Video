import numpy as np

def get_uniform_frame_indices(start_frame, stop_frame, num_frames=8, strategy='center'):
    """
    Performs uniform temporal sampling of 8 frames from a video clip.
    """
    length = stop_frame - start_frame + 1
    
    if length < num_frames:
        indices = np.linspace(start_frame, stop_frame, num_frames, dtype=int)
        return indices.tolist()
        
    segment_length = length / num_frames
    indices = []
    
    for i in range(num_frames):
        seg_start = start_frame + (i * segment_length)
        seg_end = start_frame + ((i + 1) * segment_length)
        
        if strategy == 'center':
            idx = int(seg_start + (seg_end - seg_start) / 2)
            
        elif strategy == 'random':
            idx = np.random.randint(int(seg_start), max(int(seg_start) + 1, int(seg_end)))
            
        else:
            raise ValueError("Invalid strategy. Must be 'center' or 'random'.")
            
        idx = min(max(idx, start_frame), stop_frame)
        indices.append(idx)
        
    return indices
