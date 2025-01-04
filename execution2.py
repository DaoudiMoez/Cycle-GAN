from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# Load and prepare images for testing
def load_real_samples(filename):
    """Load dataset and normalize images to the range [-1, 1]."""
    data = load(filename)
    X_blurred, X_clear = data['arr_0'], data['arr_1']
    X_blurred = (X_blurred - 127.5) / 127.5
    X_clear = (X_clear - 127.5) / 127.5
    return X_blurred, X_clear

# Select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    """Randomly select a specified number of samples from the dataset."""
    ix = randint(0, dataset.shape[0], n_samples)
    return dataset[ix]

# Display the real and generated images
def show_plot(real_images, generated_images):
    """Display real and generated images side by side."""
    images = vstack((real_images, generated_images))
    titles = ['Blurred', 'Generated']

    # Scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0

    for i in range(len(images)):
        pyplot.subplot(1, len(images), i + 1)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])
    pyplot.show()

# Main execution
if __name__ == "__main__":
    # Load dataset
    blurred_data, clear_data = load_real_samples('mini_blurred_to_normal2.npz')
    print(f'Loaded dataset: Blurred images {blurred_data.shape}, Clear images {clear_data.shape}')

    # Load the trained model
    custom_objects = {'InstanceNormalization': InstanceNormalization}
    model_AtoB = load_model('g_model_AtoB_004680.h5', custom_objects=custom_objects)

    # Select and generate images
    blurred_sample = select_sample(blurred_data, 1)
    generated_clear = model_AtoB.predict(blurred_sample)

    # Display results
    show_plot(blurred_sample, generated_clear)