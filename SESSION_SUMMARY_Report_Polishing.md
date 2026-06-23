# Session Summary — Report Polishing (June 2026)

## Purpose of This Document
This document captures every decision, text change, figure addition, structural modification, and outstanding item discussed during the report polishing session for "Project 1: Preprocessing and Bone Segmentation in Facial Ultrasound for Fracture Detection." It is intended to provide full context for any continuation chat under this project.

---

## 1. Report Generation History

### Version 1 (Claude-generated)
- Generated from project files: CLAUDE.md, progress.md, metadata_labeled.csv, and the handoff document
- Included: title page, abstract, TOC, chapters 1-8, references [1]-[8], appendices A-C
- Professional formatting with blue theme (#2E75B6), headers/footers, page numbers

### Version 2 (Claude-generated, incorporating reviews)
- Fixed Hausdorff: changed "HD95 (95th percentile)" to "Hausdorff distance (maximum, 100th percentile)" everywhere — verified that evaluate.py uses `scipy.spatial.distance.directed_hausdorff` which computes the standard maximum Hausdorff, NOT HD95
- Added abstract (was missing in v1)
- Added in-text citations [1]-[8] throughout chapters 3 and 4
- Removed raw repository paths (e.g., `runs/unet_with_augmentation_.../overlays/`) and replaced with neutral prose
- Softened "held-out patient" language to "unseen patient (not seen during training)" or "validation/test patient" with dual-use caveat stated clearly
- Title page isolated in separate document section (no header/footer/page number)
- Added TOC update note for Word users
- Bibliography reformatted with full author lists and publication details

### Version 3 (Claude-generated, adding new sections + removing UltraSAM)
- Removed all UltraSAM/Route B mentions except in Section 8.2 Recommended Improvements
- Added Section 4.3.3 "Preprocessing Alternatives Explored" (FFT, SVD, wavelet analysis)
- Added Section 4.4.1 "Bone Identification Protocol for Seed Placement" (7-point numbered protocol)
- Added reference [9] placeholder for the PCA/SVD denoising paper
- Fixed 2.4 → 2.3 renumbering for Success Criteria
- Renamed SAM2Rad from Route C to Route B throughout

### After Version 3: Manual Polishing by User
- The user took v3 as the base and has been manually editing in Word, consulting this chat for specific sections

---

## 2. Structural Changes to the Report

### Sections Removed
| Section | Reason |
|---------|--------|
| 1.3 Project Context | Redundant — title page already has institutional info; abstract mentions two-part project |
| 1.5 Report Structure (originally 1.6) | Redundant — TOC already shows the structure |
| 2.3 Scope | Redundant — scope is implicit from title, abstract, and goals |
| 6.1 Dataset Summary (table) | Redundant — duplicates the detailed dataset table in Section 4.2 |
| 6.2 Preprocessing (paragraph) | Redundant — duplicates Section 4.3; replaced with one-sentence reference |

### Sections Renumbered (after removals)
- Old 1.4 Main Challenges → **1.3**
- Old 1.5 Project Contribution → **1.4**
- Old 2.4 Success Criteria → **2.3**
- Section 6 subsections renumbered after removing 6.1 and 6.2

### Sections Reorganized
**Section 4 was significantly restructured:**
- Old 4.1 Data Description + Old 4.3.1 Input Data Handling → merged into new **4.2 Dataset and Data Acquisition**
- Old 4.2 Overall Pipeline → moved to **4.1 Pipeline Overview** (first subsection)
- Old 4.3 Preprocessing lost its 4.3.1 subsection (moved to 4.2), keeping only:
  - 4.3.1 Image Enhancement (was 4.3.2)
  - 4.3.2 Preprocessing Alternatives Explored (was 4.3.3, new content)

**New Section 4 structure:**
```
4.1 Pipeline Overview (with Figure 1 flowchart)
4.2 Dataset and Data Acquisition (with dataset composition table)
4.3 Preprocessing
    4.3.1 Image Enhancement
    4.3.2 Preprocessing Alternatives Explored
4.4 Classical Interactive Segmentation Pipeline
    4.4.1 Bone Identification Protocol for Seed Placement
    4.4.2 Seed Selection and Snapping
    4.4.3 Region Growing
    4.4.4 Morphological Refinement (renamed from "Morphological Cleaning")
    4.4.5 Active Contour Refinement
    4.4.6 Mask Output and Metadata
4.5 Dataset Indexing and Split
4.6 Deep Learning — U-Net (Route A)
    4.6.1 Architecture
    4.6.2 Training
4.7 Deep Learning — SAM2Rad (Route B)
    4.7.1 Architecture and Transfer Learning
    4.7.2 Training
```

### Section Renaming
- "Morphological Cleaning" → **"Morphological Refinement"** — because the step both removes isolated fragments AND bridges disconnected bone segments (dual role, not just cleaning)
- SAM2Rad changed from **Route C** to **Route B** throughout the entire report (U-Net remains Route A)

---

## 3. All Text Changes by Section

### Section 1.2 Problem Statement
- Added closing sentence folding in the two-project context (replacing removed Section 1.3): *"This project therefore addresses the first stage of a two-part pipeline: developing the segmentation foundation that will support subsequent 3D reconstruction and fracture analysis."*

### Section 1.4 Project Contribution
- Updated with concrete, quantified contributions (113 frames, Dice 0.678/0.671, Hausdorff percentages)

### Section 2.3 Success Criteria
- Added big-picture success criterion as opening sentence: *"At the highest level, success is defined as a working automatic bone segmentation system that generalizes to an unseen patient and produces masks reliable enough to serve as structured input for 3D reconstruction."*
- Followed by specific measurable criteria

### Section 3.7 (was "3D Ultrasound Reconstruction")
- Renamed to **"Context: Segmentation as a Foundation for Structural Analysis"**
- Shortened from a full literature review of reconstruction methods to 3 sentences
- Removed all specific reconstruction algorithm details (octrees, physics-guided DL, sensorless methods)
- Removed any mention of Project 2 specifically doing 3D reconstruction
- Kept only: motivation (segmentation enables structural analysis), Arsenescu et al. [1] as precedent, staged design justification
- Final text: "The segmentation stage addressed in this report is motivated by a broader goal: enabling automated structural analysis of facial bones from ultrasound. Prior work by Arsenescu et al. [1] demonstrated that AI-based segmentation of 2D ultrasound frames can directly support 3D reconstruction and structural comparison against CT, establishing reliable 2D mask generation as a necessary prerequisite for any downstream spatial analysis. This project focuses on that foundational segmentation stage."

### Section 3.8 Summary
- Rewrote to summarize literature review methodology choices rather than repeating 3D reconstruction context
- Removed mention of Project 2 and sensorless reconstruction
- New text focuses on: classical methods for GT generation, U-Net as baseline, SAM2Rad for transfer learning, complementary comparison

### Section 4.1 Pipeline Overview
- Moved from old 4.2 position to first subsection
- References Figure 1 (pipeline flowchart)
- Brief narrative paragraph, not a numbered list (figure does the structural work)
- Includes note that classical pipeline is the annotation tool, not the final model

### Section 4.2 Dataset and Data Acquisition
- Merged old 4.1 (Data Description) + old 4.3.1 (Input Data Handling)
- Describes: DICOM format, static vs. cine types, frame extraction protocol (equal intervals for cine), read-only raw files
- Includes detailed per-scan dataset composition table (Table 1)
- Includes metadata_labeled.csv builder description
- Patient-wise split explanation with forward reference to Section 4.5
- Dual-use limitation disclosed with forward reference to Section 7.3

### Section 4.3.1 Image Enhancement
- Reformatted from continuous paragraph to **numbered list** (3 steps: Crop, CLAHE, Gaussian blur)
- Each step has a bold label and one-sentence description with parameter values
- Followed by preprocessing strip figure (Figure 2) and CLAHE close-up figure (Figure 3)

### Section 4.3.2 Preprocessing Alternatives Explored
- Fully rebuilt with structured subsections and figures:
  1. Intro paragraph (motivation)
  2. Frequency-domain analysis + FFT spectrum figure
  3. Frequency-domain filtering + filter comparison figure (with forward reference to Section 4.4.3 for region growing)
  4. Low-rank denoising (SVD/PCA) + SVD patch comparison figure
  5. Wavelet denoising (brief, no separate figure)
  6. CNR quantitative comparison + CNR bar chart figure
  7. Conclusion
- **Critical methodological decision**: replaced the biased Dice-based preprocessing comparison (Figure D original) with an honest CNR-based comparison. Reasoning: the original Dice comparison used seeds and parameters tuned specifically for the baseline CLAHE+Gaussian preprocessing, making the baseline win by definition (Dice 0.999). CNR measures bone-to-background contrast directly from the ground-truth mask without any pipeline involvement, making it a fair comparison.
- CNR formula: |mean_bone - mean_background| / std_background
- CNR result: all methods except FFT high-pass (0.493) achieved comparable CNR (range 2.70-2.91), with differences within one standard deviation. Baseline CNR = 2.781 (ranked 7th of 11 methods, not the best).
- Honest conclusion: no alternative offered a statistically significant CNR improvement, justifying retaining the baseline without claiming it's objectively superior.
- Reference [9] added as placeholder for the PCA/SVD denoising paper — **user needs to fill in full citation**

### Section 4.4 Classical Interactive Segmentation Pipeline
- Added **4.4.1 Bone Identification Protocol for Seed Placement** — 7 numbered principles:
  1. Skip superficial layers
  2. Search for the cortical ridge
  3. Apply three confirmation criteria (continuous curve, follows anatomy, acoustic shadowing beneath)
  4. Use acoustic shadowing as supporting evidence only
  5. Evaluate brightness in context, not in isolation
  6. Use frame-to-frame consistency for cine sequences
  7. Assign confidence levels (high/medium/low)
  - Ends with bold core labeling rule
  - Note: user removed the 7th point about confidence levels from the final version (to be verified)

### Section 4.4.2 Seed Selection and Snapping
- Updated: "at least 5" → "typically 8 to 15" seeds (actual range from data)
- Added: snap_window = 7 (fixed)
- Added: seed_x_band is variable (NOT fixed at 0 as initially stated — user corrected this)
- Figure added: seed snapping figure (raw clicks red, snapped green)

### Section 4.4.3 Region Growing
- Updated with actual parameter ranges from JSON metadata:
  - tolerance: 15-20
  - seed_y_band: 0-12
  - seed_y_band_up: 0-2
  - y_band: 35 (fixed)
  - min_area: 200 (fixed)
- seed_x_band moved here from 4.4.2 as a variable spatial constraint
- Figure added: region growing result overlay

### Section 4.4.4 Morphological Refinement (renamed)
- Renamed from "Morphological Cleaning" to "Morphological Refinement"
- Text updated to describe dual role: removing speckle fragments AND bridging disconnected bone segments
- Parameters: open_r = 1 (fixed), close_w = 25-80 (variable, tuned per frame)
- Figure added: before/after morphological refinement

### Section 4.4.5 Active Contour Refinement
- Reformatted from paragraph to **numbered list of 5 components** (not sequential steps — wording changed from "applied sequentially" to "applied and configured per frame")
- Components:
  1. Pre-snake dilation (pre_snake_dilate, 0-4 px)
  2. Active contour (snake) with energy terms: alpha (0.0015-0.004), beta (0.3-0.45), w_edge=1.0, gamma=0.01, sigma=1.0
  3. Boundary smoothing (optional, sigma 1.0-1.5)
  4. Vertical trim (post_trim_up 0-3, post_trim_down 0-5)
  5. Final mask assembly (final_mask_mode: union or snake_only)
- snake_w_line = 0.0 explained as deliberate: *"The snake uses edge-based attraction exclusively (w_edge = 1.0); the intensity-based line term is disabled (w_line = 0.0) to avoid pulling the contour toward non-bone bright structures."*
- Removed: final_open_r = 0 sentence (unused parameter, not worth mentioning)
- Figure added: active contour refinement (cleaned boundary green, snake result yellow)

### Section 4.4.6 Mask Output and Metadata
- Removed final_open_r mention (unused)
- Figure added: final binary mask + overlay on original frame

### End of Section 4.4
- Gallery figure added: 3 representative frames showing seeds, final mask overlay, and per-frame parameters
- Gallery parameter boxes show only parameters that VARIED between frames
- Gallery caption lists all FIXED parameters: snap_window=7, snake_sigma=1.0, snake_gamma=0.01, snake_w_line=0.0, snake_w_edge=1.0, y_band=35, min_area=200, open_r=1

### Section 4.6.1 Architecture (U-Net)
- Added reference to architecture diagram figure
- Text trimmed to not re-describe what's visible in the diagram
- Added forward reference to Appendix B

### Section 4.6.2 Training (U-Net)
- Added forward reference to training curves in Section 6.4
- Added forward reference to Appendix B

### Section 4.7.1 Architecture and Transfer Learning (SAM2Rad)
- Fixed "two components" → **"three components"** (class tokens 2,560 params + PPN ~3M params + LoRA decoder ~3.2M params = ~6.2M total fine-tuned out of 44.1M)
- Added LoRA explanation: *"LoRA (Low-Rank Adaptation) injects small trainable low-rank matrices into the decoder's attention layers, enabling fine-tuning with far fewer parameters than full decoder retraining."*
- Added Prompt Prediction Network (PPN) as explicit third trainable component with cross-attention description
- Added total parameter context: "~6.2M fine-tuned parameters out of 44.1M total, with the remaining 38M frozen in the encoder"
- Added reference to Appendix C
- Added sentence: *"The full SAM2Rad architecture is detailed in [8]."*

### Section 4.7.2 Training (SAM2Rad)
- Added RGB conversion explanation: *"grayscale frames are repeated across three channels to match the RGB input format expected by the SA-1B-pretrained encoder"*
- Added Focal loss explanation: *"where Focal loss down-weights well-classified pixels and emphasizes hard examples, addressing the severe class imbalance between bone and background"*
- Added augmentation details (kornia-based): RandomHorizontalFlip (p=0.5), RandomAffine (rotation ±15°, translation 5%, p=0.4), RandomBrightness (factor [0.8,1.2], p=0.4), RandomContrast (factor [0.8,1.2], p=0.4)
- Added forward reference to Section 6.4 for training dynamics

### Section 5.1 Evaluation Goals
- First question reframed from "Can the proposed pipeline generate reliable automated bone masks from a two-patient training set?" to: *"Can a trained deep-learning model produce bone masks of sufficient quality — measured against the classical pipeline's masks — to serve as a scalable automatic alternative for large-volume segmentation tasks such as 3D reconstruction training data generation?"*
- Motivation: the purpose isn't to prove it works with 2 patients (that's a limitation), it's to build a scalable automatic segmentation system

### Section 5.2 Ground Truth Definition
- Changed "same parameters" to "same classical pipeline and review process... with parameters tuned per frame following the protocol in Section 4.4.1"

### Section 5.3 Quantitative Metrics
- Hausdorff diagonal normalization paragraph reframed as geometric fact: *"their image diagonals differ by a factor of two (√2·512 ≈ 724 px vs. √2·1024 ≈ 1448 px)"*

### Section 5.4 Qualitative Evaluation
- Made more specific: added "boundary accuracy, mask continuity, fragmentation, and false positive regions" to the list of aspects examined

### Section 5.5 Experimental Setup
- Added: "against identical ground-truth masks, ensuring that any performance difference reflects model capability rather than evaluation variability"

### Section 6.1 Dataset Summary
- Removed (redundant with Section 4.2). Replaced with one-sentence reference.

### Section 6.2 Preprocessing
- Reduced to one-sentence bridge referencing Section 4.3: *"The preprocessing exploration and CNR comparison are presented in Section 4.3. The baseline CLAHE + Gaussian approach was retained after no alternative showed a statistically significant improvement."*

### Section 6.3 Deep Learning Segmentation Results
- Table needs: mean ± std (not just means) for all metrics
- Table numbering: should be Table 2 (or sequential after dataset table)
- Added grouped bar chart figure comparing 4 metrics (Dice, IoU, Precision, Recall) with error bars

### Section 6.4 Training Dynamics
- Added U-Net training curve figure: two loss plots stacked vertically as (a) without augmentation and (b) with augmentation, same y-axis scale (0-0.85)
- Caption notes the loss function (0.5 × BCE + 0.5 × Dice)
- Added SAM2Rad training curve figure: loss plot only (Dice + Focal)
- Note: SAM2Rad validation computed every 20 epochs (sparse dots), unlike U-Net (every epoch)
- Best epoch discrepancy: SAM2Rad plots show epoch 60 but report text says epoch 59. User needs to verify and make consistent.

---

## 4. All Figures Added to the Report

### Pipeline and Architecture Diagrams
| Figure | Section | Description | Source |
|--------|---------|-------------|--------|
| Fig 1 | 4.1 | Pipeline overview flowchart (horizontal, color-coded: data/processing/DL) | User-made in PowerPoint |
| Fig 14 (approx) | 4.6.1 | U-Net architecture diagram (U-shape, encoder blue/bottleneck purple/decoder teal, with DoubleConv, MaxPool, ConvTranspose, 1×1 Conv → σ, input/output images) | User-made in PowerPoint |
| Fig 15 (approx) | 4.7.1 | SAM2Rad architecture diagram (horizontal flow, frozen blue/trained coral, showing Encoder → FPN → PPN → Decoder, with Class Tokens) | User-made in PowerPoint |

### Preprocessing Figures
| Figure | Section | Description |
|--------|---------|-------------|
| Fig 2 | 4.3.1 | Preprocessing strip: Original frame (with red crop box) → After crop → After CLAHE → After CLAHE + Gaussian |
| Fig 3 | 4.3.1 | Close-up comparison: cropped bone region before vs. after CLAHE enhancement |
| Fig (FFT) | 4.3.2 | Cropped ultrasound frame + 2D FFT log-magnitude spectrum (viridis colormap, cross pattern visible) |
| Fig (Filters) | 4.3.2 | Filter comparison: Original, Low-pass, High-pass, Band-pass on one frame |
| Fig (SVD) | 4.3.2 | SVD patch denoising: Original, Square 32×32, Horizontal 16×64, Vertical 64×16 |
| Fig (CNR) | 4.3.2 | CNR bar chart: mean CNR across 83 Patient 1 frames, all methods comparable except FFT high-pass (excluded), baseline in dark blue, y-axis starting at 0 |

### Classical Segmentation Figures
| Figure | Section | Description |
|--------|---------|-------------|
| Fig (seeds) | 4.4.2 | Seed snapping: raw click positions (red) vs. snapped positions (green) |
| Fig (region) | 4.4.3 | Region growing: preprocessed frame + flood-fill result overlaid (blue) |
| Fig (morph) | 4.4.4 | Morphological refinement: raw region-growing mask vs. after refinement |
| Fig (snake) | 4.4.5 | Active contour: cleaned mask boundary (green) vs. snake-refined (yellow) |
| Fig (output) | 4.4.6 | Mask output: final binary mask + overlay on original scanner frame |
| Fig (gallery) | End of 4.4 | Gallery: 3 representative frames with seeds, final mask overlay, per-frame parameter boxes |

### Results Figures
| Figure | Section | Description |
|--------|---------|-------------|
| Fig (bar chart) | 6.3 | Grouped bar chart: Dice, IoU, Precision, Recall for U-Net (blue) vs SAM2Rad (coral), with error bars |
| Fig (U-Net loss) | 6.4 | U-Net training curves: (a) without augmentation, (b) with augmentation. Loss vs epoch, same y-axis scale. Stacked vertically. |
| Fig (SAM2Rad loss) | 6.4 | SAM2Rad training loss curve (Dice + Focal). Sparse validation points (every 20 epochs). |

### Figures NOT Included (deliberately excluded)
| Figure | Reason |
|--------|--------|
| FFT bone vs. background PSD analysis | Bone-dominant band spans nearly entire frequency range — weakens the argument |
| Filter comparison with GT contour overlay (3-row version) | Decided to use single-row filter comparison instead; contour overlay approach was discussed but superseded by CNR comparison |
| SVD with prediction overlay (Dice metrics) | Too specific for preprocessing section; biased comparison issue |
| Original Dice-based preprocessing bar chart | Biased: pipeline parameters tuned for baseline, making baseline win by definition. Replaced with CNR comparison. |
| U-Net validation segmentation metrics plot (4 lines) | Too noisy and cluttered; loss curve tells the same overfitting story more clearly |
| U-Net validation Hausdorff plot | Redundant with metrics table |
| SAM2Rad Dice training curve | Redundant with loss curve (same no-overfitting story) |
| Full SAM2Rad GitHub architecture diagram | Too detailed for main text; user may include in appendix with reference to [8] |

---

## 5. All Figure Captions Written

### Section 4.1
**Figure 1.** End-to-end project pipeline, from raw DICOM data to patient-wise model evaluation.

### Section 4.2
**Table 1.** Dataset composition by patient and scan, showing the number of available frames per cine or static DICOM file and the subset selected for labeling. Patient 1 forms the training set; Patient 2 is held out for validation and final evaluation.

### Section 4.3.1
**Figure 2.** Preprocessing pipeline applied to a representative facial ultrasound frame. The original frame (left) shows the full scanner display with the crop region marked in red. Subsequent panels show the result after spatial cropping (600 × 600 px), CLAHE contrast enhancement (clip limit 0.01), and Gaussian blur (7 × 7 kernel). The bright curved ridge visible in the upper portion of the cropped frames corresponds to the hyperechoic cortical bone interface targeted by segmentation.

**Figure 3.** Close-up of the cortical bone region before and after CLAHE enhancement, showing the increased prominence and local contrast of the hyperechoic bone ridge.

### Section 4.3.2
**Figure (FFT).** A representative cropped ultrasound frame (left) and its 2D FFT log-magnitude spectrum (right).

**Figure (Filters).** Frequency-domain filter outputs on a representative frame: original, low-pass, high-pass, and band-pass.

**Figure (SVD).** SVD patch denoising with three patch shapes: square (32×32), horizontal (16×64), and vertical (64×16).

**Figure (CNR).** Mean CNR comparison across all 83 labeled Patient 1 frames. FFT high-pass (CNR = 0.493) is excluded due to its substantially lower score; all remaining methods fall within one standard deviation of the baseline.

### Section 4.4
**Figure (seeds).** Raw click positions (red) and snapped positions (green) for a representative frame.

**Figure (region growing).** Preprocessed frame (left) and the combined flood-fill result overlaid in blue (right).

**Figure (morphological refinement).** Raw region-growing mask (left) and after morphological refinement (right), showing gap bridging between disconnected bone segments.

**Figure (active contour).** Cleaned mask boundary before snake refinement (green, left) and after active contour refinement (yellow, right).

**Figure (mask output).** Final binary mask (left) and its overlay on the original scanner frame (right).

**Figure (gallery).** Three representative labeled frames showing seed placements (left), final mask overlays (center), and per-frame parameter values (right). Parameters fixed across all labeled frames are omitted from the parameter boxes and listed in the caption: snap_window = 7, snake_sigma = 1.0, snake_gamma = 0.01, snake_w_line = 0.0, snake_w_edge = 1.0, y_band = 35, min_area = 200, open_r = 1.

### Section 4.6.1
**Figure (U-Net arch).** U-Net architecture used in Route A. The encoder contracts spatial resolution via MaxPool while increasing channel depth. The decoder expands via ConvTranspose2d with skip-connected encoder features concatenated at each level. A 1×1 convolution with sigmoid activation produces the final single-channel output. Total trainable parameters: ~1.9M.

### Section 4.7.1
**Figure (SAM2Rad arch).** SAM2Rad architecture used in Route B. The frozen SAM2 image encoder (Hiera Tiny, 38M parameters) produces multi-scale FPN features. Learnable class tokens are cross-attended against these features by the Prompt Prediction Network (PPN), producing sparse and dense prompt embeddings. The LoRA-adapted mask decoder receives both the image embedding and the predicted prompts to produce the final segmentation mask. Only coral components are fine-tuned (~6.2M parameters total); the encoder remains frozen throughout training. The full internal architecture is detailed in [8]. Total: 44.1M params (38M frozen + 6.2M fine-tuned).
- **Note:** Caption needs fixing — says "Only coral components" but class tokens are shown in purple in the user's diagram. Should say "Only non-blue components are trained" or list both coral and purple.

### Section 6.3
**Figure (metrics bar).** Comparison of segmentation metrics on the 30 Patient 2 test frames. Error bars show ±1 standard deviation across frames. Overlapping error bars on all four metrics indicate no statistically significant difference between the two models.

### Section 6.4
**Figure (U-Net loss).** U-Net training and validation loss curves (loss = 0.5 × BCE + 0.5 × Dice). (a) Without augmentation: a clear train/val gap develops after ~30 epochs, indicating overfitting (best epoch 36). (b) With augmentation: the gap is nearly eliminated, confirming that augmentation was critical for generalization (best epoch 47).

**Figure (SAM2Rad loss).** SAM2Rad training and validation segmentation loss (Dice + Focal). Training and validation loss track closely throughout, with no meaningful gap developing — consistent with the transfer-learning design where only lightweight components are fine-tuned. Validation was computed every 20 epochs. Best checkpoint at epoch 59.

---

## 6. Key Decisions Made

### Hausdorff Metric
- evaluate.py uses `scipy.spatial.distance.directed_hausdorff` → standard maximum Hausdorff (100th percentile), NOT HD95
- All references to "HD95" were corrected to "Hausdorff distance"
- The metric is sensitive to boundary outliers (single stray pixel can inflate the value)
- Diagonal normalization values (724 px, 1448 px) are geometric facts (Pythagorean theorem on image dimensions), not data-specific measurements

### UltraSAM (Route B → removed)
- UltraSAM was a planned comparison route that was never implemented due to environment complexity (OpenMMLab framework, Python 3.8 requirement, COCO data conversion)
- Decision: remove ALL mentions from the report except one sentence in Section 8.2 Recommended Improvements
- In 8.2, phrased as: "Evaluate UltraSAM as an ultrasound-specific pretrained segmentation model, which may offer stronger domain transfer than the general SA-1B-pretrained SAM2 backbone."
- No "Route B" label used for UltraSAM anymore; SAM2Rad takes the Route B designation

### Route Naming
- **Route A** = U-Net (trained from scratch)
- **Route B** = SAM2Rad (SAM2 transfer learning)
- Route C designation no longer exists

### Preprocessing Comparison Methodology
- The original Dice-based comparison was identified as biased: seeds and parameters were tuned for the baseline, making it win by definition (Dice 0.999)
- Replaced with CNR (Contrast-to-Noise Ratio) comparison: |mean_bone - mean_background| / std_background
- CNR computed using ground-truth masks to define bone/background regions — no pipeline involvement
- Result: baseline is NOT the best by CNR (ranked 7th/11), but all methods are statistically comparable (differences within 1 std)
- Honest conclusion: no alternative significantly outperforms baseline, justifying retaining it without claiming superiority
- An honest comparison would have required either: (a) re-tuning pipeline for each method, (b) training separate DL models per method, or (c) CNR/SNR evaluation — option (c) was chosen as the feasible and fair approach

### Section 4.4.4 Naming
- Renamed from "Morphological Cleaning" to "Morphological Refinement"
- Reason: the step serves a dual role — removing speckle AND connecting disconnected bone segments. "Cleaning" implies only removal.

### Parameters: Defaults vs. Actuals
- The original report cited "default" parameter values that didn't represent actual usage
- Decision: replace all defaults with actual ranges from the JSON metadata files
- Fixed parameters (identical across all frames): snap_window=7, snake_sigma=1.0, snake_gamma=0.01, snake_w_line=0.0, snake_w_edge=1.0, y_band=35, min_area=200, open_r=1
- Variable parameters (tuned per frame): num_seeds (8-15), tolerance (15-20), seed_x_band (variable), seed_y_band (0-12), seed_y_band_up (0-2), close_w (25-80), pre_snake_dilate (0-4), snake_alpha (0.0015-0.004), snake_beta (0.3-0.45), boundary_smooth (True/False), boundary_smooth_σ (1.0-1.5), post_trim_up (0-3), post_trim_down (0-5), final_mask_mode (union/snake_only)
- seed_x_band was initially listed as fixed at 0 but user corrected: it varies across frames
- final_open_r = 0 was unused and removed from the text entirely
- snake_w_line = 0.0 kept with explanation: deliberate choice to use edge-based attraction only

### SAM2Rad Architecture
- The text originally said "two components are fine-tuned" but the architecture actually has THREE trainable components:
  1. Class tokens (2,560 params)
  2. Prompt Prediction Network / PPN (~3M params)
  3. LoRA-adapted mask decoder (~3.2M params)
- Text corrected to "three components"
- Full architecture description available from Claude Code output (stored in user's project)
- Detailed ChatGPT-generated diagram available — suggested for appendix, not main text

### Training Curves
- U-Net: include only loss curves (not segmentation metrics or Hausdorff — too noisy/redundant)
- Show both runs (without/with augmentation) stacked vertically as (a)/(b) with matched y-axis scales
- SAM2Rad: include only loss curve (not Dice — redundant)
- SAM2Rad validation is sparse (every 20 epochs) — noted in caption
- Best epoch discrepancy for SAM2Rad: plots show epoch 60, text says 59. User needs to verify.

### 3D Reconstruction Context
- Section 3.7 was shortened to avoid implying Project 2 will specifically be 3D reconstruction
- The user does not want to commit to what Project 2 will contain
- Section frames segmentation as a "foundation for structural analysis" without specifying next steps

---

## 7. Outstanding Items / Still TODO

### Must Fix Before Submission
1. **Reference [9]**: The PCA/SVD denoising paper citation is a placeholder — needs full author list, journal, volume, year
2. **SAM2Rad best epoch**: Plots show epoch 60, report text says epoch 59. Verify which is correct and make consistent across text, table, appendix, and plots
3. **Patient 2 Scan 3 (0 labeled frames)**: Either label the frames before submission, remove the row from the dataset table, or add a footnote explaining why it was excluded
4. **SAM2Rad figure caption**: Says "Only coral components are fine-tuned" but class tokens are purple in the user's diagram. Fix to "Only non-blue components are trained" or mention both colors
5. **Metrics table**: Add ± std values to all metrics (Dice, IoU, Precision, Recall, Hausdorff) — compute from per-sample CSVs
6. **SAM2Rad input label in figure**: Shows "512×512" which is the U-Net input size. Should show 600×600 crop resized to 1024×1024
7. **Pipeline flowchart**: Fix "Rout A"/"Rout B" typos to "Route A"/"Route B". Add Evaluation color to legend (currently coral but not in legend)

### Sections Still Needing Full Review
8. **Section 6.5 Qualitative Observations**: Still has placeholder sentence "Overlay figures are generated automatically for all 30 test frames and should be inserted here for the final submission." Needs actual overlay comparison figures (success + failure cases)
9. **Section 7 Discussion**: Not yet reviewed in this polishing session
10. **Section 8 Conclusions and Future Work**: Not yet reviewed
11. **Appendices A, B, C**: Not yet reviewed against the updated parameter information. Appendix A parameter table should match the actual ranges from Section 4.4. Need to verify consistency.
12. **Appendix D (optional)**: Could include the detailed ChatGPT-generated SAM2Rad diagram

### Nice-to-Have Improvements
13. **Section 6.5**: Add 2-3 success case overlays and 2-3 failure case overlays as comparison figures (Original | U-Net overlay | SAM2Rad overlay)
14. **Table numbering**: Verify all tables are numbered sequentially and consistently referenced in text
15. **Figure numbering**: Assign final figure numbers to all figures and update all text references
16. **Appendix B**: Verify U-Net training configuration matches the updated text in 4.6.2
17. **Appendix C**: Verify SAM2Rad configuration matches the updated text in 4.7.1-4.7.2 (especially the three-component description and augmentation details)
18. **Abstract**: May need minor updates to reflect any changes made during polishing (e.g., Route B instead of Route C)

---

## 8. Report Table of Contents (Current Final Structure)

```
Abstract

1. Introduction
   1.1 Clinical and Engineering Motivation
   1.2 Problem Statement
   1.3 Main Challenges
   1.4 Project Contribution

2. Project Goals and Scope
   2.1 Main Objectives
   2.2 Specific Tasks
   2.3 Success Criteria

3. Background and Literature Review
   3.1 Facial Bone Imaging and Fracture Detection
   3.2 Ultrasound Imaging of Bone
   3.3 Preprocessing in Ultrasound Imaging
   3.4 Classical Bone Segmentation Methods
   3.5 Deep Learning Segmentation: U-Net
   3.6 Foundation Models for Medical Segmentation
   3.7 Context: Segmentation as a Foundation for Structural Analysis
   3.8 Summary

4. Materials and Methods
   4.1 Pipeline Overview
   4.2 Dataset and Data Acquisition
   4.3 Preprocessing
       4.3.1 Image Enhancement
       4.3.2 Preprocessing Alternatives Explored
   4.4 Classical Interactive Segmentation Pipeline
       4.4.1 Bone Identification Protocol for Seed Placement
       4.4.2 Seed Selection and Snapping
       4.4.3 Region Growing
       4.4.4 Morphological Refinement
       4.4.5 Active Contour Refinement
       4.4.6 Mask Output and Metadata
   4.5 Dataset Indexing and Split
   4.6 Deep Learning — U-Net (Route A)
       4.6.1 Architecture
       4.6.2 Training
   4.7 Deep Learning — SAM2Rad (Route B)
       4.7.1 Architecture and Transfer Learning
       4.7.2 Training

5. Evaluation Methodology
   5.1 Evaluation Goals
   5.2 Ground Truth Definition
   5.3 Quantitative Metrics
   5.4 Qualitative Evaluation
   5.5 Experimental Setup

6. Results
   6.1 [Removed — reference Section 4.2]
   6.2 Preprocessing [one-sentence bridge to Section 4.3]
   6.3 Deep Learning Segmentation Results
   6.4 Training Dynamics
   6.5 Qualitative Observations

7. Discussion
   7.1 Interpretation of Results
   7.2 Strengths of the Proposed Pipeline
   7.3 Limitations
   7.4 Implications for Project 2
   7.5 Comparison to Initial Expectations

8. Conclusions and Future Work
   8.1 Main Conclusions
   8.2 Recommended Improvements
   8.3 Continuation to Project 2

9. References [1]-[9]

Appendix A — Classical Pipeline Parameters
Appendix B — U-Net Training Configuration
Appendix C — SAM2Rad Configuration
Appendix D — [Optional] Detailed SAM2Rad Architecture Diagram
```

---

## 9. References List (Current State)

```
[1] Arsenescu, V., et al. (2023). "3D Ultrasound Reconstructions of the Carotid Artery and Thyroid Gland Using AI-Based Automatic Segmentation." Sensors.
[2] Victoria, C., et al. (2023). "Real-Time 3D Ultrasound Reconstruction Using Octrees." IEEE Access.
[3] Dou, Y., et al. (2024). "Sensorless End-to-End Freehand 3D Ultrasound Reconstruction with Physics-Guided Deep Learning." IEEE Trans. UFFC.
[4] Solberg, O. V., et al. (2011). "3D Ultrasound Reconstruction Algorithms from Analog and Digital Data." Ultrasonics.
[5] Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
[6] Kirillov, A., et al. (2023). "Segment Anything." ICCV.
[7] Ravi, N., et al. (2024). "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714. Meta AI.
[8] Wahd, A. "SAM2Rad: Adapting SAM2 for Radiology." GitHub: github.com/aswahd/SamRadiology.
[9] [PLACEHOLDER — NEEDS FULL CITATION] "Multiresolution Generalized N-Dimensional PCA for Ultrasound Image Denoising."
```

---

## 10. Color Theme and Styling Decisions

### Report Document
- Blue theme (#2E75B6) for headings, table headers, header/footer borders
- Table style: dark blue header row with white text, alternating light blue (#F2F7FC) / white rows, thin gray borders
- All tables should use the same style throughout

### Pipeline Flowchart (Figure 1)
- Light green: data nodes
- Light blue: processing/labeling nodes
- Light green: deep learning model nodes (U-Net, SAM2Rad)
- Light coral/salmon: evaluation node
- Dashed box: "Automatic segmentation learning" container
- Note: Evaluation color (coral) was missing from legend — needs fixing

### U-Net Architecture Diagram (Figure 14)
- Light blue: encoder blocks
- Light purple: bottleneck
- Light green: decoder blocks
- Light peach: output head (1×1 Conv, Sigmoid)
- Input/output: actual ultrasound frame and binary mask images

### SAM2Rad Architecture Diagram (Figure 15)
- Blue: frozen components (Image Encoder)
- Purple: learnable prompts (Class Tokens)
- Coral/orange: fine-tuned components (PPN, Mask Decoder)
- Input/output: actual ultrasound frame and binary mask images

### Charts and Plots
- U-Net: blue bars/lines
- SAM2Rad: coral/orange bars/lines
- Baseline preprocessing: dark blue bar
- Alternative preprocessing: light blue bars
- Error bars: black
- Best epoch: dashed vertical line

---

## 11. Files Referenced in the Project

### Code Files
- `ultrasound_bone_segmentation_cli.py` — classical segmentation tool
- `build_metadata_labeled.py` — metadata CSV builder
- `dcm_to_png_batch.py` — DICOM extraction script
- `run_patient1_by_index.py` — batch processing script
- `UNet/evaluate.py` — U-Net evaluation (uses scipy.spatial.distance.directed_hausdorff)
- `SAM2Rad/evaluate.py` — SAM2Rad evaluation

### Data Files
- `Dataset/metadata_labeled.csv` — 113 entries, master index
- `Dataset/Patient1/Masks/*.json` — per-mask parameter metadata
- `Dataset/Patient1/IMG_frames/` — PNG frames
- `Dataset/Patient1/DCM_frames/` — raw DICOMs (read-only)

### Output/Results Files
- `runs/unet_with_augmentation_20260531_143449/` — best U-Net run
- `runs/sam2rad_bone_seg_eval/` — SAM2Rad evaluation
- Per-sample metric CSVs in each run's eval directory
- CNR comparison: `cnr_per_frame.csv`, `cnr_summary.csv`
