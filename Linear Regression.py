import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from PIL import Image

# Fix random seed for reproducibility
np.random.seed(42)

# --- STEP 1: Generate synthetic object data ---
# Let's assume 30 objects, sizes in cm
num_objects = 30
real_heights = np.random.uniform(10, 50, num_objects)  # cm
real_widths  = np.random.uniform(10, 40, num_objects)
real_lengths = np.random.uniform(15, 60, num_objects)

# Camera scaling ratios (pixels per cm), slightly varied for realism
# Updated camera scaling ratios to match toy car's image-based scale
scale_height = np.random.normal(23.54, 0.03, num_objects)
scale_width  = np.random.normal(24.167, 0.03, num_objects)
scale_length = np.random.normal(24.167, 0.03, num_objects)

# Simulated pixel sizes captured by 3 orthogonal cameras
pixel_heights = real_heights * scale_height
pixel_widths  = real_widths  * scale_width
pixel_lengths = real_lengths * scale_length

# --- STEP 2: Create DataFrame ---
df = pd.DataFrame({
    'pixel_height': pixel_heights,
    'pixel_width': pixel_widths,
    'pixel_length': pixel_lengths,
    'real_height': real_heights,
    'real_width': real_widths,
    'real_length': real_lengths
})

# --- STEP 3: Train Linear Regression Models ---
height_model = LinearRegression().fit(df[['pixel_height']], df['real_height'])
width_model  = LinearRegression().fit(df[['pixel_width']],  df['real_width'])
length_model = LinearRegression().fit(df[['pixel_length']], df['real_length'])

# --- STEP 4: Predict new object size from pixels ---
# Simulate a new object
test_real_height = 13.34  # cm
test_real_width  = 15.62  # cm
test_real_length = 18.00  # cm

# Load the image
image_path = '/home/hp/WhatsApp Image 2025-04-16 at 14.11.26.jpeg'  # Replace with actual image path
img = Image.open(image_path)

# Convert to numpy array
img_array = np.array(img)

# Get image height and width
height, width, _ = img_array.shape

# Split the image into left (top view) and right (side view)
left_view = img_array[:, :width//2, :]
right_view = img_array[:, width//2:, :]

# Function to find bounding box of the object
def find_bounding_box(view):
    # Create a mask where pixels are not white (background)
    # Using a threshold to account for near-white pixels
    mask = np.any(view < [250, 250, 250], axis=2)
    coords = np.argwhere(mask)
    
    # Bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

# Get dimensions from top view
x_min_l, y_min_l, x_max_l, y_max_l = find_bounding_box(left_view)
width_top = x_max_l - x_min_l
length_top = y_max_l - y_min_l

# Get dimensions from side view
x_min_r, y_min_r, x_max_r, y_max_r = find_bounding_box(right_view)
length_side = x_max_r - x_min_r
height_side = y_max_r - y_min_r

# Width from top view
width = width_top  

# Length is consistent - use the one from top view
length = length_top  

# Height from side view
height = height_side 

test_pixel_height = height   # from image
test_pixel_width  = width
test_pixel_length = length


# Predict
predicted_height = height_model.predict([[test_pixel_height]])[0]
predicted_width  = width_model.predict([[test_pixel_width]])[0]
predicted_length = length_model.predict([[test_pixel_length]])[0]

print("Predicted Size (cm):")
print(f"  Height: {predicted_height:.2f} cm")
print(f"  Width:  {predicted_width:.2f} cm")
print(f"  Length: {predicted_length:.2f} cm")

print("\nActual Size (cm):")
print(f"  Height: {test_real_height:.2f} cm")
print(f"  Width:  {test_real_width:.2f} cm")
print(f"  Length: {test_real_length:.2f} cm")

# --- (Optional) Plot regression fit ---
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].scatter(df['pixel_height'], df['real_height'], color='blue', label='Data')
axs[0].plot(df['pixel_height'], height_model.predict(df[['pixel_height']]), color='red')
axs[0].set_title('Height Regression')
axs[0].set_xlabel('Pixel Height')
axs[0].set_ylabel('Real Height (cm)')

axs[1].scatter(df['pixel_width'], df['real_width'], color='green')
axs[1].plot(df['pixel_width'], width_model.predict(df[['pixel_width']]), color='red')
axs[1].set_title('Width Regression')
axs[1].set_xlabel('Pixel Width')
axs[1].set_ylabel('Real Width (cm)')

axs[2].scatter(df['pixel_length'], df['real_length'], color='purple')
axs[2].plot(df['pixel_length'], length_model.predict(df[['pixel_length']]), color='red')
axs[2].set_title('Length Regression')
axs[2].set_xlabel('Pixel Length')
axs[2].set_ylabel('Real Length (cm)')

plt.tight_layout()
plt.show()

