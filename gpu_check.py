#!/usr/bin/env python3
import os


def _env_indicates_no_gpu():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        return False
    value = cuda_visible.strip()
    return value == "" or value == "-1"


def gpu_available(use_tensorflow=False):
    """
    Fast GPU availability check.
    - If CUDA_VISIBLE_DEVICES disables GPU, return False immediately.
    - If use_tensorflow is False, do not import TensorFlow.
    - If use_tensorflow is True, use TensorFlow to detect GPUs.
    """
    if _env_indicates_no_gpu():
        return False
    if not use_tensorflow:
        return False
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except Exception:
        return False


def main():
    try:
        import tensorflow as tf
    except Exception as e:
        print("=== GPU Configuration Check ===")
        print(f"TensorFlow import failed: {e}")
        print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set')}")
        return

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
        except Exception:
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
            print("\nGPU computation test: SUCCESS")
            print(f"Result: {c.numpy()}")
    except Exception:
        print("\nGPU computation test: FAILED")


if __name__ == "__main__":
    main()
