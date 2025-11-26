# AI Agent Instructions for 3D-US-Reconstruction Project

This project focuses on 3D ultrasound reconstruction for facial fracture detection, with emphasis on bone segmentation using various techniques. Here's what you need to know to work effectively with this codebase:

## Project Architecture

The project is organized into major components:
- `3D Volume reconstruction/` - Volume reconstruction from US images
- `Bone Segmentation/` - Core segmentation algorithms
- `Estimating Probe Pose/` - Probe position tracking
- `Pre-processing/` - Image preprocessing utilities
- `Post-processing/` - Post-segmentation refinement

### Key Implementation Patterns

1. **Bone Segmentation Pipeline**:
   - Images go through preprocessing → segmentation → post-processing
   - Main implementation in `Bone Segmentation/Region Growing Segmentation/seg/ultrasound_bone_segmentation.py`
   - Uses combined region growing and active contour approach

2. **Image Processing Workflow**:
   ```python
   # Standard preprocessing chain
   img = load_image(path)
   img = crop_ultrasound_region(img)
   img = preprocess_image(img)  # CLAHE + Gaussian blur
   ```

3. **Segmentation Parameters**:
   - Region growing tolerance: typically 10-15
   - Active contour parameters: 
     ```python
     alpha=0.0005, beta=2, gamma=0.01, w_line=0.5, w_edge=0.0
     ```

## Development Workflows

1. **Image Processing Development**:
   - Test new algorithms on single images first
   - Use interactive region growing for parameter tuning
   - Validate results visually using matplotlib visualization

2. **Region Growing Development**:
   - Support both manual and automatic seed point selection
   - Multiple seeds can be combined using logical OR operations
   - Clean results using morphological operations

## Critical Dependencies

- OpenCV (`cv2`) for basic image operations
- scikit-image for advanced segmentation algorithms
- matplotlib for visualization and interactive tools
- NumPy for array operations

## Integration Points

1. **Data Flow**:
   - Input: Raw ultrasound images (PNG/JPEG)
   - Output: Segmentation masks (PNG)
   - Intermediate: Preprocessed images, contours

2. **Interactive Components**:
   - Manual seed selection using matplotlib events
   - Visual validation of segmentation results

## Best Practices

1. Always validate image loading and preprocessing:
   ```python
   if img is None:
       raise FileNotFoundError(f"Could not load image: {path}")
   ```

2. Use proper image normalization before processing:
   ```python
   img = img / 255.0  # Normalize to [0,1]
   ```

3. Implement clear cleanup and error handling for user interactions