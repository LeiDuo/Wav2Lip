import enum

# 可选配置的默认值
default_config = {
    # If True, then use only first video frame for inference
    'static': False,
    # Can be specified only if input is a static image (default: 25)
    'fps': 25,
    # Padding (top, bottom, left, right). Please adjust to include chin at least
    'pads': [0, 10, 0, 0],
    # Batch size for face detection
    'face_det_batch_size': 16,
    # Batch size for Wav2Lip model(s)
    'wav2lip_batch_size': 128,
    # Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p
    'resize_factor': 1,
    # Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg.
    # Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
    'crop': [0, -1, 0, -1],
    # Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.
    # Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
    'box': [-1, -1, -1, -1],
    # Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.
    # Use if you get a flipped result, despite feeding a normal looking video
    'rotate': False,
    # Prevent smoothing face detections over a short temporal window
    'nosmooth': False
}
