import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def create_test_images(output_dir="test_images"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    w, h = 512, 512
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)

    # ──────────────────────────────────────────────────────────
    # 1. The "All-in-One" (Gradients, Shapes, Noise, Bump)
    # ──────────────────────────────────────────────────────────
    print("Generating 1_all_in_one.png...")
    r = (X / w * 255).astype(np.uint8)
    g = (Y / h * 255).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    img = np.stack([r, g, b], axis=-1)

    # Gaussian bump
    cx, cy = w // 2, h // 2
    gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 80**2)) * 255
    img[..., 0] = np.clip(img[..., 0].astype(np.float64) + gauss * 0.5, 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(img[..., 1].astype(np.float64) + gauss * 0.8, 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(img[..., 2].astype(np.float64) + gauss * 1.0, 0, 255).astype(np.uint8)

    # Shapes
    img[50:150, 50:150] = [220, 50, 50]  # Red square
    mask = (X - 400)**2 + (Y - 100)**2 < 60**2
    img[mask] = [50, 220, 50]             # Green circle
    
    # Noise block
    noise = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img[350:450, 350:450] = noise

    plt.imsave(f"{output_dir}/1_all_in_one.png", img)

    # ──────────────────────────────────────────────────────────
    # 2. High Contrast (Great for Thresholding & Edge Detection)
    # ──────────────────────────────────────────────────────────
    print("Generating 2_high_contrast.png...")
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    
    # White background with dark shapes
    img2[100:400, 100:400] = [240, 240, 240] 
    img2[150:350, 150:350] = [20, 20, 20]    # Black square
    mask = (X - 250)**2 + (Y - 250)**2 < 70**2
    img2[mask] = [240, 240, 240]             # White circle inside
    
    # Add a bright red rectangle
    img2[50:80, 400:480] = [255, 0, 0]
    
    plt.imsave(f"{output_dir}/2_high_contrast.png", img2)

    # ──────────────────────────────────────────────────────────
    # 3. Low Light / Underexposed (Test Brightness/Gamma adjustments)
    # ──────────────────────────────────────────────────────────
    print("Generating 3_low_light.png...")
    # Dark base with slight noise
    img3 = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    
    # Add some dim features
    mask1 = (X - 150)**2 + (Y - 150)**2 < 80**2
    img3[mask1] = [30, 40, 60] # Slight blue tint
    
    mask2 = (X - 350)**2 + (Y - 350)**2 < 80**2
    img3[mask2] = [60, 30, 30] # Slight red tint
    
    # One bright spot
    mask3 = (X - 250)**2 + (Y - 250)**2 < 15**2
    img3[mask3] = [200, 200, 200]
    
    plt.imsave(f"{output_dir}/3_low_light.png", img3)

    # ──────────────────────────────────────────────────────────
    # 4. Color Wheel (Test HSV Heatmaps & Eigen Color Clusters)
    # ──────────────────────────────────────────────────────────
    print("Generating 4_color_wheel.png...")
    # Create HSV color wheel
    angle = np.arctan2(Y - h/2, X - w/2)
    hue = (angle + np.pi) / (2 * np.pi) # 0 to 1
    sat = np.sqrt((X - w/2)**2 + (Y - h/2)**2) / (w/2)
    sat = np.clip(sat, 0, 1)
    val = np.ones_like(hue)
    
    hsv = np.stack([hue, sat, val], axis=-1)
    img4 = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    
    # Cut out a black background outside the circle
    outside = sat > 1.0
    img4[outside] = [0, 0, 0]
    
    plt.imsave(f"{output_dir}/4_color_wheel.png", img4)

    # ──────────────────────────────────────────────────────────
    # 5. Concentric Ripples (Amazing for Height Map 3D view)
    # ──────────────────────────────────────────────────────────
    print("Generating 5_ripples_heightmap.png...")
    dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
    ripples = np.cos(dist / 8.0) # Create wave pattern
    
    # Map -1..1 to 50..255 (keeping it slightly bright for LUT mapping)
    ripples_mapped = ((ripples + 1) / 2 * 205 + 50).astype(np.uint8)
    
    img5 = np.stack([ripples_mapped, ripples_mapped, ripples_mapped], axis=-1)
    # Add a subtle color gradient overlay
    img5[..., 0] = np.clip(ripples_mapped.astype(int) + (X / w * 50 - 25).astype(int), 0, 255).astype(np.uint8)
    img5[..., 2] = np.clip(ripples_mapped.astype(int) + (Y / h * 50 - 25).astype(int), 0, 255).astype(np.uint8)
    
    plt.imsave(f"{output_dir}/5_ripples_heightmap.png", img5)

    # ──────────────────────────────────────────────────────────
    # 6. Plaid / Crosshatch (Great for Profiles and Filters)
    # ──────────────────────────────────────────────────────────
    print("Generating 6_crosshatch.png...")
    pattern_x = np.sin(X / 15.0)
    pattern_y = np.sin(Y / 25.0)
    plaid = (pattern_x + pattern_y)
    
    # Map to 0-255
    plaid_mapped = ((plaid + 2) / 4 * 255).astype(np.uint8)
    
    img6 = np.stack([
        plaid_mapped, 
        np.clip(plaid_mapped.astype(int) - 40, 0, 255).astype(np.uint8), 
        np.clip(plaid_mapped.astype(int) + 40, 0, 255).astype(np.uint8)
    ], axis=-1)
    
    plt.imsave(f"{output_dir}/6_crosshatch.png", img6)

    print(f"\n✅ Successfully generated 6 test images in the '{output_dir}/' folder!")
    print("Load them into the Image Analyzer and click '⚡ Analyze All'")

if __name__ == "__main__":
    create_test_images()