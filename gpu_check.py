#!/usr/bin/env python3
import tensorflow as tf
import os

print("=== GPU Configuration Check ===")

# Check CUDA availability
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")

# List physical devices
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        
    # Check GPU memory
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU Details: {gpu_details}")
    except:
        print("Could not get GPU details")
else:
    print("No GPUs detected by TensorFlow")

# Check environment variables
print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}")

# Test GPU computation
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"\nGPU computation test: SUCCESS")
        print(f"Result: {c.numpy()}")
except:
    print("\nGPU computation test: FAILED")