import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# Step 1: Preprocessing
def preprocess_image(image_path, is_sketch):
    """
    Preprocesses the input image by applying edge detection or binary thresholding.

    Args:
        image_path (str): Path to the image file.
        is_sketch (bool): True if the image is a sketch, False for a test image.

    Returns:
        np.ndarray: Cropped region of the image containing valid edges or binary thresholded zones.
    """
    # Read the grayscale version of the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verify if the image was loaded correctly
    if img is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")

    if is_sketch:
        # Apply binary thresholding to emphasize the sketch details
        otsu_threshold, _ = cv2.threshold(img, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        _, edges = cv2.threshold(img, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # For test images, smooth the image with Gaussian blur
        img = cv2.GaussianBlur(img, (15, 15), 1.5)
        # Apply Canny edge detection to extract edges
        otsu_threshold, _ = cv2.threshold(img, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        edges = cv2.Canny(img, threshold2=(0.5 * otsu_threshold), threshold1=otsu_threshold)

    # Extract the valid zone containing meaningful edges
    valid_zone = _GetValidZone(edges)

    return valid_zone


def histogram_computation(valid_zone, W, K, th_edge_ratio):
    """
    Computes the histograms for image blocks based on gradient orientations.

    Args:
        valid_zone (np.ndarray): Preprocessed image with valid edge zones.
        W (int): Number of spatial blocks along one dimension (granularity of spatial division).
        K (int): Number of bins in the histogram (granularity of orientation).
        th_edge_ratio (float): Minimum edge ratio threshold for considering a block.

    Returns:
        np.ndarray: Filtered histograms for the image blocks.
    """
    # Divide the valid zone into blocks of size W x W
    image_blocks = _DivideImage(valid_zone, W)

    # Compute Sobel gradients for each block
    Gx, Gy = _BlockSobelGradient(image_blocks)

    # Compute denoised gradient orientations for each block
    alpha_blocks = _BlockDenoisedOrientation(Gx, Gy)

    # Compute histograms based on the orientations
    histogram_blocks = _Histogram(alpha_blocks, K)

    # Filter out blocks that do not meet the edge threshold
    filtered_histogram_blocks = _FilterBlocks(histogram_blocks, image_blocks, th_edge_ratio)

    return filtered_histogram_blocks


def _FilterBlocks(histogram, image_blocks, threshold_edge_ratio):
    """
    Filters histogram blocks by removing blocks with insufficient edge information.

    Args:
        histogram (np.ndarray): Histograms of orientation angles for the blocks.
        image_blocks (np.ndarray): Blocks of the valid zone.
        threshold_edge_ratio (float): Minimum ratio of edges required to retain a block.

    Returns:
        np.ndarray: Filtered histogram blocks.
    """
    h_block_num, w_block_num = histogram.shape
    filtered_histogram = np.zeros((h_block_num, w_block_num))

    # Retain blocks only if their edge ratio exceeds the threshold
    for i in range(h_block_num):
        for j in range(w_block_num):
            if np.sum(image_blocks[i, j]) > (image_blocks[i, j].size * threshold_edge_ratio):
                filtered_histogram[i, j] = histogram[i, j]

    return filtered_histogram


def _Histogram(alpha_blocks, K):
    """
    Computes the histogram of orientations for image blocks.

    Args:
        alpha_blocks (np.ndarray): Gradient orientations for each block.
        K (int): Number of orientation bins in the histogram.

    Returns:
        np.ndarray: Histograms of gradient orientations for the blocks.
    """
    # Normalize orientations into K bins
    hist_blocks = alpha_blocks / (np.pi / K)
    hist_blocks = hist_blocks.astype(np.int32) * np.pi / K
    return hist_blocks


def _GetValidZone(data):
    """
    Extracts the valid zone from the preprocessed image containing meaningful edges.

    Args:
        data (np.ndarray): Binary edge-detected image.

    Returns:
        np.ndarray: Cropped region containing valid edges.
    """
    count_h, count_w = np.sum(data, axis=1), np.sum(data, axis=0)
    idx_valid_h, idx_valid_w = np.argwhere(count_h > 0), np.argwhere(count_w > 0)
    y_low, y_high = np.min(idx_valid_h), np.max(idx_valid_h)
    x_low, x_high = np.min(idx_valid_w), np.max(idx_valid_w)
    return data[y_low:y_high + 1, x_low:x_high + 1]


def _BlockDenoisedOrientation(Gx, Gy):
    """
    Computes denoised orientations from Sobel gradients.

    Args:
        Gx, Gy (np.ndarray): Gradients along the X and Y directions for image blocks.

    Returns:
        np.ndarray: Denoised gradient orientations.
    """
    h_block_num, w_block_num, _, _ = Gx.shape
    Lx, Ly = np.zeros((h_block_num, w_block_num)), np.zeros((h_block_num, w_block_num))

    # Compute Lx and Ly as aggregated gradient metrics
    for i in range(h_block_num):
        for j in range(w_block_num):
            Ly[i, j] = 2 * np.sum(Gx[i, j] * Gy[i, j])
            Lx[i, j] = np.sum(Gx[i, j] ** 2 - Gy[i, j] ** 2)

    # Apply Gaussian smoothing to reduce noise in orientations
    sigma, window_size = 0.5, 3
    Lx = cv2.GaussianBlur(Lx, (window_size, window_size), sigma)
    Ly = cv2.GaussianBlur(Ly, (window_size, window_size), sigma)

    # Compute the gradient orientations (alpha)
    alpha = 0.5 * (np.arctan2(Ly, Lx) + np.pi)
    return alpha


def compute_similarity(hist1, hist2, method="manhattan"):
    """
    Computes similarity between two histograms using a chosen distance metric.

    Args:
        hist1, hist2 (np.ndarray): Histograms to be compared.
        method (str): Similarity metric; options are "manhattan" or "chi_square".

    Returns:
        float: Similarity score between the histograms.
    """
    if method == "chi_square":
        eps = 1e-10
        return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    elif method == "manhattan":
        return np.sum(np.abs(hist1 - hist2))
    else:
        raise ValueError("Unknown method specified for similarity calculation.")

def _DivideImage(data, W):
    """
    Divides the input image into blocks for processing.

    Args:
        data (np.ndarray): The input valid zone or edge-detected image.
        W (int): The number of blocks along each dimension (granularity).

    Returns:
        np.ndarray: Divided image blocks of size (W x W).
    """
    h, w = data.shape
    h_block_size, w_block_size = h // W, w // W
    image_blocks = np.zeros((W, W, h_block_size, w_block_size))

    # Divide the image into non-overlapping blocks
    for i in range(W):
        for j in range(W):
            image_blocks[i, j, :, :] = data[h_block_size * i:h_block_size * (i + 1),
                                            w_block_size * j:w_block_size * (j + 1)]
    return image_blocks


def _BlockSobelGradient(data_blocks):
    """
    Computes the Sobel gradients for each image block.

    Args:
        data_blocks (np.ndarray): Divided image blocks.

    Returns:
        tuple: Gradients along the X-axis (Gx) and Y-axis (Gy).
    """
    h_block_num, w_block_num, _, _ = data_blocks.shape
    Gx, Gy = np.zeros(data_blocks.shape), np.zeros(data_blocks.shape)

    # Compute Sobel gradients for each block
    for i in range(h_block_num):
        for j in range(w_block_num):
            block_data = data_blocks[i, j]
            Gx[i, j] = cv2.Sobel(block_data, cv2.CV_64F, 1, 0, ksize=3)
            Gy[i, j] = cv2.Sobel(block_data, cv2.CV_64F, 0, 1, ksize=3)

    return Gx, Gy


def choose_K_based_on_resolution(sketch_image):
    """
    Dynamically determines the number of orientation bins (K) based on the sketch resolution.

    Args:
        sketch_image (np.ndarray): The input sketch image.

    Returns:
        int: The dynamically determined number of bins (K).
    """
    height, width = sketch_image.shape
    scaling_factor = 1000  # Determines sensitivity to resolution
    return min(180, max(36, (height * width) // scaling_factor))


def choose_W_based_on_resolution(sketch_image):
    """
    Dynamically determines the number of spatial blocks (W) based on the sketch resolution.

    Args:
        sketch_image (np.ndarray): The input sketch image.

    Returns:
        int: The dynamically determined number of blocks (W).
    """
    height, width = sketch_image.shape
    block_size = 32  # Desired block size in pixels
    return max(1, min(height, width) // block_size)


def process_test_image(test_image_path, sketch_histo, W, K, th_edge_ratio):
    """
    Processes a single test image and computes its similarity to the sketch.

    Args:
        test_image_path (str): Path to the test image.
        sketch_histo (np.ndarray): Histogram of the sketch image.
        W (int): Number of spatial blocks.
        K (int): Number of orientation bins.
        th_edge_ratio (float): Minimum edge ratio threshold.

    Returns:
        tuple: Test image path and its similarity score.
    """
    # Preprocess the test image (apply edge detection and cropping)
    test_edges = preprocess_image(test_image_path, is_sketch=False)

    # Compute histograms for the test image
    test_histo = histogram_computation(test_edges, W, K, th_edge_ratio)

    # Compute similarity between the sketch and test image histograms
    similarity = compute_similarity(sketch_histo, test_histo)

    return test_image_path, similarity


def rank_images(query_image_path, auto=True, K=72, W=25, test_image_dir="../dataset/", method="chi_square",
                th_edge_ratio=0.5):
    """
    Ranks test images based on their similarity to the query sketch.

    Args:
        query_image_path (str): Path to the query sketch image.
        test_image_dir (str): Path to the directory containing test images.
        method (str): Similarity metric ("manhattan" or "chi_square").
        th_edge_ratio (float): Minimum edge ratio threshold for valid blocks.
        auto (bool): If True, automatically determine K and W based on the sketch resolution.
        K (int): Default number of orientation bins.
        W (int): Default number of spatial blocks.

    Returns:
        list: Top 3 most similar test images and their similarity scores.
    """
    # Preprocess the query sketch image
    sketch_preprocessed = preprocess_image(query_image_path, is_sketch=True)

    # Dynamically determine K and W if auto is enabled
    if auto:
        K = choose_K_based_on_resolution(sketch_preprocessed)
        W = choose_W_based_on_resolution(sketch_preprocessed)

    print(K)
    print(W)
    print(auto)

    # Compute histograms for the sketch
    sketch_histo = histogram_computation(sketch_preprocessed, W, K, th_edge_ratio)

    # Collect paths of all test images
    test_image_paths = []
    for folder_path in os.listdir(test_image_dir):
        folder_path = os.path.join(test_image_dir, folder_path)
        for test_image_name in os.listdir(folder_path):
            test_image_path = os.path.join(folder_path, test_image_name)
            if test_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image_paths.append(test_image_path)

    # Process all test images in parallel using multithreading
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(
            lambda path: process_images(path, sketch_histo, th_edge_ratio, K, W),
            test_image_paths
        )
        similarity_scores = list(results)

    # Sort test images by similarity scores
    similarity_scores.sort(key=lambda x: x[1])

    # Display and rank the top 3 most similar test images if the script is run as the main module
    if __name__ == "__main__":
        print("Ranking of Test Images (Most Similar First):")
        for rank, (image_path, score) in enumerate(similarity_scores[:3], start=1):
            print(f"Rank {rank}: {os.path.basename(image_path)} (Similarity Score: {score})")
            img = cv2.imread(image_path)
            plt.subplot(1, 3, rank)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Rank {rank}\nScore: {score}")
            plt.axis('off')
        plt.show()

    # Return the top 3 results
    return similarity_scores[:3]



def process_images(test_image_path, sketch_histo, th_edge_ratio_test, K, W):
    """
    Processes a test image and computes its similarity to the query sketch.

    Args:
        test_image_path (str): Path to the test image.
        sketch_histo (np.ndarray): Histogram of the sketch image.
        th_edge_ratio_test (float): Minimum edge ratio threshold.
        K (int): Number of orientation bins.
        W (int): Number of spatial blocks.

    Returns:
        tuple: Test image path and its similarity score.
    """
    # Preprocess the test image
    test_image = preprocess_image(test_image_path, is_sketch=False)

    # Compute histograms for the test image
    test_histo = histogram_computation(test_image, W, K, th_edge_ratio_test)

    # Compute similarity between the sketch and test image histograms
    similarity_score = compute_similarity(sketch_histo, test_histo)

    return test_image_path, similarity_score


if __name__ == "__main__":
    """
    Main execution: Compare a query sketch to a set of test images and rank them.
    """
    query_image_path = "../sketch/sketch2.png"  # Path to the query sketch
    rank_images(query_image_path,True,None,None)
