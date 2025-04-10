import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

tf.config.optimizer.set_jit(True)
os.makedirs('../generated_images', exist_ok=True)

filters = tf.constant([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=tf.float32)
filters = tf.reshape(filters, [3, 3, 1, 1])


lambda_sobel = 0.01  # just like lambda_curv before
lambda_grad = 0.0
lambda_curv = 0.0
lambda_tv = 0.01
lambda_elevation = 0.0


def build_generator(noise_dim, output_shape=(256, 256, 4)):
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
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(noise_dim,)))

    # Normalise the output of the previous layer, scaling the activations to stabilise the training process, helping the network converge faster and avoid issues like vanishing and exploding gradients.
    # Re-enabling BatchNormalization
    model.add(layers.BatchNormalization())

    # Introduces Rectified Linear Unit activation function, which allows a small, non-zero gradient when the layer output is negative - helps avoid dead neurons which improves gradient flow.
    model.add(layers.LeakyReLU())

    # Reshape the output into a 16x16x256 feature map
    model.add(layers.Reshape((4, 4, 1024)))

    # Layer 2: Transforms low-dimensional input from the dense layer into a higher-dimensional one for image-like representation.
    # 128 represents the number of filters, outputting 128 feature maps. Each map represents a distinct set of features that the model will learn, (5, 5) is the kernel size, meaning each filter is a 5x5 grid.
    # stride value means that filter will move one pixel at a time vertically and horizontally across the input, padding means that output will have same dimensions as input, and bias against removes bias.
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    # Re-enabling BatchNormalization
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Layer 3: Another upsampling layer, similar settings to previous layer, however the number of filters is halved since it doesn't need them to capture complex patterns in the layer. The initial layer needs more filters to learn abstract features.
    # The stride value change is the key difference, it is doubles, meaning that we are doubling the spatial dimensions, upsampling to increase the width and height of the output
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    # Re-enabling BatchNormalization
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Added more layers for upsampling
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
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

    # added more layers for downsampling
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and output a single probability for real or fake classification
    # Flatten takes multidimensional output from the previous layer to convert it to a 1D vector to linearises the data, suitable for passing into a dense layer.
    model.add(layers.Flatten())
    # Dense layer has a single output neuron with a sigmoid activation function, producing value between 0 and 1; the probability that the input is real (1) or fake (0).
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Defining loss function. BinaryCrossentropy loss function distinguishes between two classes, real or fake images.
# from_logits=False because the discriminator's output is passed through a sigmoid activation.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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

    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    real_labels = tf.ones_like(real_output) * 0.95  # Label smoothing
    fake_labels = tf.zeros_like(fake_output) + 0.05  # Label smoothing for fake labels
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss

# Introduced learning rate decay
initial_lr = 0.0004  # or your current learning rate

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000,      # every 10,000 steps
    decay_rate=0.95,        # reduce by 5%
    staircase=True          # decay in steps (not smooth)
)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

# Step 5: Training Step with RGB+DEM Data
def train_step(rgb_dem_images, generator, discriminator, noise_dim):
    """
    A single training step for the GAN.
    Parameters:
        rgb_dem_images (tf.Tensor): Batch of real RGB+DEM images.
        generator (tf.keras.Model): The Generator model.
        discriminator (tf.keras.Model): The Discriminator model.
        noise_dim (int): Dimension of the noise vector input for the Generator.
    """
    print("Batch shape in train_step:", rgb_dem_images.shape)

    # Ensure the input images are in the float32 format
    rgb_dem_images = tf.cast(rgb_dem_images, tf.float32)

    # Check for NaN or Inf in real images
    tf.debugging.check_numerics(rgb_dem_images, "Real images contain NaN or Inf")

    # Create a batch of random noise vectors
    noise = tf.random.normal([rgb_dem_images.shape[0], noise_dim])

    # Gradient tape is a tool for recording operations that happen within its context so that it can calculate gradients of a given function with respect to its input.
    # After operations are complete, TensorFlow can use the recorded information to calculate the derivative of a function (like loss) and these are used to adjust the model's parameters.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Passes the created noise to the generator to create fake images, sets training to true which affect Batch Normalisation and dropout layers
        generated_images = generator(noise, training=True)

        # Check for NaN or Inf in generated images
        tf.debugging.check_numerics(generated_images, "Generated images contain NaN or Inf")

        # Adding noise to discriminator inputs to prevent it from becoming too confident
        noise_factor = 0.01
        real_images_noisy = rgb_dem_images + noise_factor * tf.random.normal(shape=rgb_dem_images.shape)
        generated_images_noisy = generated_images + noise_factor * tf.random.normal(shape=generated_images.shape)

        # Discriminator evaluates both real and fake images by giving a value from 0 to 1 as per the sigmoid activation function
        real_output = discriminator(real_images_noisy, training=True)
        fake_output = discriminator(generated_images_noisy, training=True)

        # **Check for NaN or Inf in discriminator outputs**
        tf.debugging.check_numerics(real_output, "Discriminator output on real images contains NaN or Inf")
        tf.debugging.check_numerics(fake_output, "Discriminator output on fake images contains NaN or Inf")

        # Saves the return of the loss functions
        gen_loss = generator_loss(fake_output)
        # Custom loss components
        real_dem = rgb_dem_images[:, :, :, 3:4]  # Extract real DEM channel
        fake_dem = generated_images[:, :, :, 3:4]  # Extract fake DEM channel

        # Compute structure-aware losses separately
        sobel = sobel_loss(real_dem, fake_dem)
        grad = gradient_loss(real_dem, fake_dem)

        # Log raw structure losses
        tf.print("Sobel Loss:", sobel)
        tf.print("Gradient Loss:", grad)

        # Add structure-aware losses
        gen_loss += lambda_grad * grad
        gen_loss += lambda_sobel * sobel
        gen_loss += lambda_tv * total_variation_loss(fake_dem)

        # Elevation distribution penalty
        target_elevation_mean = 0.5
        mean_fake_elevation = tf.reduce_mean(fake_dem)
        elevation_penalty = tf.abs(mean_fake_elevation - target_elevation_mean)
        gen_loss += lambda_elevation * elevation_penalty

        # Add structure-aware losses
        gen_loss += lambda_grad * gradient_loss(real_dem, fake_dem)
        gen_loss += lambda_curv * sobel_loss(real_dem, fake_dem)
        gen_loss += lambda_tv * total_variation_loss(fake_dem)

        # Elevation distribution penalty
        target_elevation_mean = 0.5
        mean_fake_elevation = tf.reduce_mean(fake_dem)
        elevation_penalty = tf.abs(mean_fake_elevation - target_elevation_mean)
        gen_loss += lambda_elevation * elevation_penalty

        disc_loss = discriminator_loss(real_output, fake_output)

        # **Check for NaN or Inf in losses**
        tf.debugging.check_numerics(gen_loss, "Generator loss contains NaN or Inf")
        tf.debugging.check_numerics(disc_loss, "Discriminator loss contains NaN or Inf")

    # .gradient() calculates the gradients of gen_loss with respect to the generator's trainable variables.
    # A gradient is the derivative of the loss function with respect to each weight. It shows how much the loss would increase or decrease if we change a specific weight.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Re-enabling gradient clipping to prevent exploding gradients
    gradients_of_generator, _ = tf.clip_by_global_norm(gradients_of_generator, 10.0)
    # apply_gradients() updates the generator's weights by applying these gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Same for discriminator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Re-enabling gradient clipping for discriminator
    gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 10.0)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Compute gradient norms for logging
    gen_grad_norm = tf.linalg.global_norm(gradients_of_generator)
    disc_grad_norm = tf.linalg.global_norm(gradients_of_discriminator)
    print(f"Generator gradient norm: {gen_grad_norm}")
    print(f"Discriminator gradient norm: {disc_grad_norm}")
    print(f"Mean fake DEM elevation: {mean_fake_elevation:.4f}")

    return gen_loss, disc_loss, gen_grad_norm.numpy(), disc_grad_norm.numpy()


def train(dataset, epochs, generator, discriminator, noise_dim):
    print("Started training")
    saved_models_folder = "saved_models"
    os.makedirs(saved_models_folder, exist_ok=True)
    best_loss = float('inf')
    patience = 20  # epochs to wait before stopping
    wait = 0

    gen_grad_norms = []
    disc_grad_norms = []

    gen_losses = []
    disc_losses = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_gen_losses = []
        epoch_disc_losses = []

        for step, rgb_dem_batch in enumerate(dataset):
            # Visualize a batch of training data (only in the first epoch and step)
            if epoch == 0 and step == 0:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for i in range(min(rgb_dem_batch.shape[0], 4)):
                    img = rgb_dem_batch[i].numpy()
                    # RGB channels
                    axes[0, i].imshow(img[:, :, :3])
                    axes[0, i].set_title(f"Input RGB {i + 1}")
                    axes[0, i].axis('off')
                    # DEM channel
                    axes[1, i].imshow(img[:, :, 3], cmap='terrain')
                    axes[1, i].set_title(f"Input DEM {i + 1}")
                    axes[1, i].axis('off')
                plt.tight_layout()
                plt.savefig(f"training_input_epoch_{epoch + 1}_step_{step + 1}.png")
                plt.close()

            # Perform a single training step
            gen_loss, disc_loss, gen_grad_norm, disc_grad_norm = train_step(rgb_dem_batch, generator, discriminator,
                                                                            noise_dim)
            epoch_gen_losses.append(gen_loss.numpy())
            epoch_disc_losses.append(disc_loss.numpy())
            gen_grad_norms.append(gen_grad_norm)
            disc_grad_norms.append(disc_grad_norm)

        avg_gen_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        avg_disc_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)


        # Force save every 15 epochs, regardless of improvement
        if (epoch + 1) % 15 == 0:
            generate_and_save_images(generator, epoch + 1, noise_dim)
            forced_model_path = get_unique_filename(saved_models_folder, "generator_forced_epoch", epoch + 1)
            generator.save(forced_model_path)

            # Also save the arrays you track (e.g. gen_losses, disc_losses, grad_norms, etc.)
            np.save(f"generator_losses_epoch_{epoch + 1}.npy", np.array(gen_losses))
            np.save(f"discriminator_losses_epoch_{epoch + 1}.npy", np.array(disc_losses))
            np.save(f"generator_grad_norms_epoch_{epoch + 1}.npy", np.array(gen_grad_norms))
            np.save(f"discriminator_grad_norms_epoch_{epoch + 1}.npy", np.array(disc_grad_norms))

    print("Training completed.")


# A way to visualise the progress of the model throughout training
def generate_and_save_images(model, epoch, noise_dim):
    # Generate a batch of images
    noise = tf.random.normal([16, noise_dim])
    generated_images = model(noise, training=False)

    # Debug output for the first image in the batch
    img_debug = generated_images[0].numpy()

    # Visualize RGB Channels
    fig_rgb, axes_rgb = plt.subplots(4, 4, figsize=(8, 8))
    fig_rgb.suptitle(f'Epoch {epoch} - RGB Channels', fontsize=16)

    for i, ax in enumerate(axes_rgb.flat):
        img = generated_images[i].numpy()

        # Extract RGB channels and rescale from [-1, 1] to [0, 1]
        rgb_image = (img[:, :, :3] + 1) / 2.0

        for c in range(3):  # Loop through R, G, B channels
            p2, p98 = np.percentile(rgb_image[:, :, c], (2, 98))
            print(f"Generated RGB {i + 1} Band {c + 1} Percentiles: p2={p2}, p98={p98}")
            rgb_image[:, :, c] = np.clip((rgb_image[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)

        ax.imshow(rgb_image)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('../generated_images', f'generated_images_epoch_{epoch}_rgb.png'))
    plt.close(fig_rgb)

    # Visualize DEM Channel
    fig_dem, axes_dem = plt.subplots(4, 4, figsize=(8, 8))
    fig_dem.suptitle(f'Epoch {epoch} - DEM Channel', fontsize=16)

    for i, ax in enumerate(axes_dem.flat):
        img = generated_images[i].numpy()

        dem_image = (img[:, :, 3] + 1) / 2.0

        ax.imshow(dem_image, cmap='terrain')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('../generated_images', f'generated_images_epoch_{epoch}_dem.png'))
    plt.close(fig_dem)


def generate_full_resolution_image(generator, noise_dim, output_filename):
    # Generating one large image
    noise = tf.random.normal([1, noise_dim])  # Single image
    generated_image = generator(noise, training=False)

    # Converting from [-1,1] to [0,1]
    generated_image = (generated_image.numpy()[0] + 1) / 2.0

    # Saving RGB and DEM separately
    rgb_image = generated_image[:, :, :3]
    dem_image = generated_image[:, :, 3]

    # Save as npy
    # np.save(output_filename + '_rgb.npy', rgb_image)
    # np.save(output_filename + '_dem.npy', dem_image)

    # png
    plt.imsave(output_filename + '_rgb.png', rgb_image)
    plt.imsave(output_filename + '_dem.png', dem_image, cmap='gray')


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
    data = []

    for f in files:
        arr = np.load(f).astype(np.float32)

        # Normalize RGB from [0, 255] to [-1, 1]
        arr[:, :, :3] = (arr[:, :, :3] / 127.5) - 1.0

        # Normalize DEM from [0, 1000] to [-1, 1]
        arr[:, :, 3] = (arr[:, :, 3] / 500.0) - 1.0

        data.append(arr)

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


def get_unique_filename(base_path, base_name, epoch=None):
    """
    Generates a filename, optionally tagged with the epoch number.

    Parameters:
        base_path (str): Path to the folder where the file will be saved.
        base_name (str): Base name for the file (without extension).
        epoch (int, optional): If provided, include it in the filename.

    Returns:
        str: A file path including the epoch or a unique index.
    """
    os.makedirs(base_path, exist_ok=True)

    if epoch is not None:
        file_path = os.path.join(base_path, f"{base_name}_epoch_{epoch}.h5")
        return file_path
    else:
        index = 0
        file_path = os.path.join(base_path, f"{base_name}.h5")
        while os.path.exists(file_path):
            index += 1
            file_path = os.path.join(base_path, f"{base_name}_{index}.h5")
        return file_path


def gradient_loss(real, fake):
    real_dx, real_dy = tf.image.image_gradients(real)
    fake_dx, fake_dy = tf.image.image_gradients(fake)
    return tf.reduce_mean(tf.abs(real_dx - fake_dx)) + tf.reduce_mean(tf.abs(real_dy - fake_dy))

def sobel_loss(real, fake):
    real_edges = tf.image.sobel_edges(real)
    fake_edges = tf.image.sobel_edges(fake)
    return tf.reduce_mean(tf.abs(real_edges - fake_edges))

def total_variation_loss(img):
    return tf.reduce_mean(tf.image.total_variation(img))

if __name__ == "__main__":
    # Dimension of the noise vector
    noise_dim = 100
    output_shape = (256, 256, 4)

    batch_size = 64
    epochs = 2000

    # Build the generator and discriminator
    generator = build_generator(noise_dim, output_shape)
    discriminator = build_discriminator(output_shape)

    # Load dataset with RGB+DEM data pairs
    output_folder_path = "/workspace/preprocessed_data_resized"
    dataset = load_dataset(output_folder_path, batch_size)

    for batch in dataset.take(1):

        plt.figure(figsize=(8, 8))
        for i in range(min(batch.shape[0], 4)):
            img = batch[i].numpy()
            plt.subplot(1, min(batch.shape[0], 4), i + 1)
            plt.imshow((img[:, :, :3] + 1) / 2.0)
            plt.axis('off')
        plt.show()

    # Start training
    train(dataset, epochs, generator, discriminator, noise_dim)

    # Save generator model in a unique filename under 'saved_models' folder
    saved_models_folder = "saved_models"
    generator_model_path = get_unique_filename(saved_models_folder, "generator_model")
    generator.save(generator_model_path)
    print(f"Generator model saved to: {generator_model_path}")
