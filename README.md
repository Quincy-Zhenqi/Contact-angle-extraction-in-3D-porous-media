# 3D Contact Angle Extraction from Segmented  TIFF Stacks

![figure_abstract_1](https://github.com/user-attachments/assets/4f1259a4-b36d-43cf-b73e-136d18209b52)


This repository provides a Python code to extract a **3D contact-angle distribution** from **segmented** three-phase volumetric TIFF images using **geometric plane fitting** and multiple **physics/quality constraints**.

**Phase labels (required):**
- **1 = solid**
- **2 = liquid**
- **3 = gas**

Please refer to the image data we have provided.

The script outputs:
- Contact-angle list and summary statistics
- Histogram figure + histogram CSV
- Per-point CSV (XYZ + angle + local phase ratios)
- A 3D **contact-angle voxel map TIFF** with the **same shape as the input** (angle in degrees at valid triple-phase points; 0 elsewhere)
- An `.npz` bundle containing intermediate arrays (good/bad points, vectors, ratio filters, etc.)

---


## Requirements

### Python
- **Python 3.9+** recommended (3.8+ should work)

### Dependencies
- `numpy`
- `tifffile`
- `matplotlib`
- `scipy`
- `pandas`

### Citation

If you use this code in academic work, please cite the associated manuscript:

Guo et al. (2025), Physics-constrained contact angle extraction in 3D porous media

DOI：xxxx

