import tensorflow as tf

# source tf_env/bin/activate

# Check for a list of available physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Physical devices:", physical_devices)

if physical_devices:
    print("\nGPU is detected. TensorFlow is now ready to use Metal acceleration.")
else:
    print("\nGPU was not detected.")
    
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print("\nMatched a tensor on the gpu. Result:\n", c)
except RuntimeError as e:
    print("\nTensorFlow is not configured to run on the GPU. Error: ")
    print(e)