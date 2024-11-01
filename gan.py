import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def build_generator(noise_dim, output_shape):
    """
    Builds the Generator model that transforms a noise vector into synthetic RGB+DEM data.
    Parameters:
        noise_dim (int): Dimension of the noise vector input.
        output_shape (tuple): The shape of the output. Ex: (28, 28, 4) for RGB+DEM channels.
    Returns:
        tf.keras.Model: Compiled Generator model.
    """
    # Keras: high-level neural network API built on top of Tensorflow. Simplifies process of
    # creating and training deep learning models. Used to build models in a modular way adding layers sequentially.
    # Sequential is a type of Keras model where layers are stacked in a linear way from input to output, step by step. Each layer feeds its output directly into the next layer.
    model = tf.keras.Sequential()

    # Layer 1: Begin with a dense(fully connected) layer, which transform 1D noise input (a random vector), into a higher-dimensional representation.
    # A dense layer connects every neuron in the current layer to every neuron in the following layer. Each connection has a weight associated with it, which the model learns and adjusts during training.
    # A neuron (aka node/unit) is a fundamental processing unit that takes one or more inputs, applied a weight to each input, adds a bias term and then passes the result through and activation function to produce and output.
    # Each neuron in a layer receives inputs from neurons in the previous layer, for each input, it applies a weight, which is a value that determines the strength or importance of that input to the neuron's output. These are learned through training.
    # A bias term is added to the weighted sum, which gives the model more flexibility to fit the data by shifting the output, and the weighted sum and bias are passed through and activation function, such as ReLU, which help introduce non-linearity into the network, allowing it to model complex relationships.
    # First, we add a dense layer as defined above and 7*7*256 is the number of neurons in the layer, a form that can be upscaled into an image. No bias is used, and the input shape is the vector of a random noise - starting point of the generator.
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))

    # Normalise the output of the previous layer, scaling the activations to stabilise the training process, helping the network converge faster and avoid issues like vanishing and exploding gradients.
    model.add(layers.BatchNormalization())

    # Introduces Rectified Linear Unit activation function, which allows a small, non-zero gradient when the layer output is negative - helps avoid dead neurons which improves gradient flow.
    model.add(layers.LeakyReLU())

    # Reshape the output into a 7x7x256 feature map
    model.add(layers.Reshape((7, 7, 256)))

    # Layer 2: Transforms low-dimensional input from the dense layer into a higher-dimensional one for image-like representation.
    # 128 represents the number of filters, outputting 128 feature maps. Each map represents a distinct set of features that the model will learn, (5, 5) is the kernel size, meaning each filter is a 5x5 grid.
    # stride value means that filter will move one pixel at a time vertically and horizontally across the input, padding means that output will have same dimensions as input, and bias against removes bias.
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Layer 3: Another upsampling layer, similar settings to previous layer, however the number of filters is halved since it doesn't need them to capture complex patterns in the layer. The initial layer needs more filters to learn abstract features.
    # The stride value change is the kay difference, it is doubles, meaning that we are doubling the spatial dimensions, upsampling to increase the width and height of the output
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Layer 4: Final upsampling to match the desired output shape for RGB and DEM (RGB 3 channels + DEM 1 channel = 4 channels)
    model.add(layers.Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator(input_shape):
    """
    Builds the Discriminator model that distinguishes real RGB+DEM data from fake ones.
    Parameters:
        input_shape (tuple): Shape of the input image, e.g., (28, 28, 4).
    Returns:
        tf.keras.Model: Compiled Discriminator model.
    """
    model = tf.keras.Sequential()

    # Layer 1: Convolutional layer to downsample the input image, increasing spatial dimensions to form an image.
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU())
    # Used dropout to randomly ignore 30% of the neurons during training, preventing overfitting by reducing reliance on any specific neuron.
    model.add(layers.Dropout(0.3))

    # Layer 2: Another convolutional layer with increased filters. Upsamples the image, expanding spatial dimensions to increase resolution.
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and output a single probability for real or fake classification
    # Flatten takes multidimensional output from the previous layer to convert it to a 1D vector to linearises the data, suitable for passing into a dense layer.
    model.add(layers.Flatten())
    # Dense layer has a single output neuron with a sigmoid activation function, producing value between 0 and 1; the probability that the input is real (1) or fake (0).
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Defining loss function. BinaryCrossentropy loss function distinguishes between two classes, real or fake images.
# from_logits is true because input to loss function is not passed through a sigmoid activation, since we are using it directly in discriminator's final layer.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    """
    Loss function for the Generator.
    It measures how well the Generator 'tricks' the Discriminator into thinking generated images are real.

    Parameters:
        fake_output (tf.Tensor): Discriminator's classifications of the Generator's images.

    Returns:
        tf.Tensor: Loss for the Generator.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    """
    Loss function for the Discriminator.
    It measures how well the Discriminator distinguishes real images from fake ones.

    Parameters:
        real_output (tf.Tensor): Discriminator's classification of real images.
        fake_output (tf.Tensor): Discriminator's classification of fake images.

    Returns:
        tf.Tensor: Loss for the Discriminator.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Setting up optimisers: adjusts model's parameters to minimise loss function. Uses Adaptive moment estimation(Adam) to adapt learning rate dynamically for each parameter.
# Learning rate: 1e-4 = 0.0001 specifies the step size which is the amount by which weights are updates with each training iteration
# If it's too high the model becomes unstable, too low it would take a long time or get stuck in a local minimum.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Step 5: Training Step with RGB+DEM Data
@tf.function
def train_step(rgb_dem_images, generator, discriminator, noise_dim):
    """
    A single training step for the GAN.
    Parameters:
        rgb_dem_images (tf.Tensor): Batch of real RGB+DEM images.
        generator (tf.keras.Model): The Generator model.
        discriminator (tf.keras.Model): The Discriminator model.
        noise_dim (int): Dimension of the noise vector input for the Generator.
    """
    # Create a batch of random noise vectors
    noise = tf.random.normal([rgb_dem_images.shape[0], noise_dim])

    # Gradient tape is a tool for recording operations that happen within its context so that it can calculate gradients of a given function with respect to its input.
    # After operations are complete, TensorFlow can use the recorded information to calculate the derivative of a function (like loss) and these are used to adjust the model's parameters.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Passes the created noise to the generator to create fake images, sets training to true which affect Batch Normalisation and dropout layers
        generated_images = generator(noise, training=True)

        # Discriminator evaluates both real and fake images by giving a value from 0 to 1 as per the sigmoid activation function
        real_output = discriminator(rgb_dem_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Saves the return of the loss functions
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # .gradient() calculates the gradients of gen_loss with respect to the generator's trainable variables.
    # A gradient is the derivative of the loss function with respect to each weight. It shows how much the loss would increase or decrease if we change a specific weight.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # apply_gradients() updates the generator's weights by applying these gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Same for discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, generator, discriminator, noise_dim):
    # Loop over all epochs: one epoch is one full pass through the entire dataset
    for epoch in range(epochs):
        # Within each epoch, we divide the dataset into small chunks called batches
        # For each batch, we perform the training step which updates the generator and discriminator and after each epoch we generate and save the images
        for rgb_dem_batch in dataset:
            train_step(rgb_dem_batch, generator, discriminator, noise_dim)
        generate_and_save_images(generator, epoch + 1, noise_dim)

# A way to visualise the progress of the model throughout training
def generate_and_save_images(model, epoch, noise_dim):
    # Generate a batch of images
    noise = tf.random.normal([16, noise_dim])

    # Set training  = false for inference mode
    generated_images = model(noise, training=False)

    # Set up the plot for a 4x4 grid of generated images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'Epoch {epoch}', fontsize=16)

    for i, ax in enumerate(axes.flat):
        # Extract image from the batch and remove the last channel if grayscale
        img = generated_images[i].numpy()

        # Clip image to range [0, 1] for display: rescale from [-1, 1] to [0, 1]
        img = (img + 1) / 2.0

        # Display the image: handle grayscale and RGB+DEM (if needed, split channels here)
        ax.imshow(img[:, :, :3])

        # Hide axes
        ax.axis('off')

    # Save the figure to a file
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.show()
    plt.close(fig)


def load_dataset(output_folder_path, batch_size):
    """
    Loads preprocessed RGB+DEM data from .npy files in a folder and prepares it as a TensorFlow dataset.

    Parameters:
        output_folder_path (str): Path to the folder containing preprocessed .npy files.
        batch_size (int): Number of samples per batch.

    Returns:
        tf.data.Dataset: Prepared dataset for training.
    """
    # Load all .npy files and create a dataset
    files = sorted([os.path.join(output_folder_path, f) for f in os.listdir(output_folder_path) if f.endswith('.npy')])
    data = [np.load(f) for f in files]

    # Convert to a numpy array and then to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset

if __name__ == "__main__":
    # Dimension of the noise vector
    noise_dim = 100
    # For example, 28x28 with 3 RGB channels + 1 DEM channel
    output_shape = (28, 28, 4)

    batch_size = 32
    epochs = 50

    # Build the generator and discriminator
    generator = build_generator(noise_dim, output_shape)
    discriminator = build_discriminator(output_shape)

    # Load dataset with RGB+DEM data pairs
    output_folder_path = "/workspace/preprocessed_data"
    dataset = load_dataset(output_folder_path, batch_size)

    # Start training
    train(dataset, epochs, generator, discriminator, noise_dim)
