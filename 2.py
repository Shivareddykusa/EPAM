from PIL import Image

# Load your unenhanced image
unenhanced_image_path = "/path/to/your/unenhanced/image.jpg"
unenhanced_image = plt.imread(unenhanced_image_path)
unenhanced_image = unenhanced_image.astype('float32') / 255.0
unenhanced_image = np.expand_dims(unenhanced_image, axis=0)

# Use the trained autoencoder to enhance the image
enhanced_image = autoencoder.predict(unenhanced_image)

# Display the original and enhanced images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(unenhanced_image.squeeze(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Enhanced Image')
plt.imshow(enhanced_image.squeeze(), cmap='gray')

plt.show()
