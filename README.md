# Using GAN to Generate 3D Video Game Terrain

> *Generate  RGB+DEM terrain tiles—and stitch them into full maps—using a custom Deep Convolutional GAN.*

---

## 1. Project Overview

This repository contains the full pipeline developed for my final‑year project. Starting from raw satellite imagery and elevation data exported from Google Earth Engine (GEE), it trains a DCGAN that produces 4‑channel outputs (RGB + height). A seam‑aware blending routine then mosaics many small tiles into a single, large, seamless landscape suitable for GIS or real‑time graphics.


---

## 2. Pipeline at a Glance

| Step            | Script                                | Purpose                                 |
| --------------- | ------------------------------------- | --------------------------------------- |
| **Extract**     | `extract_earth_engine.js`             | Export `big_rgb.tif` & `big_dem.tif` from GEE |
| **Tile**        | `grid_data.py`                        | Split rasters into 512×512 tiles        |
| **Validate**    | `validate_grid_data.py`               | Ensure every RGB tile has a matching DEM |
| **Pre‑process** | `pre‑process.py`                      | Normalise, resize→256², augment         |
| **Train GAN**   | `gan.py`                              | Learn joint RGB+DEM distribution        |
| **Generate**    | `create_terrain.py`                   | Create single blocks *or* stitched maps |


*All intermediate artefacts live under `output_tiles/` and `preprocessed_data/` , however did not fit for submission.*

---

## 3. Environment

| Component  | Version                                                 |
| ---------- | ------------------------------------------------------- |
| Python     | 3.10                                                    |
| TensorFlow | 2.14 (GPU)                                              |
| CUDA       | 11.8                                                    |
| cuDNN      | 8.6                                                     |
| Core libs  | NumPy · Rasterio · Matplotlib · SciPy · Pillow · psutil |

### 3.1 Quick Docker (recommended)

```bash
# 1– Run an interactive GPU container, mounting the repo
$ docker run --gpus all -it `
  -v "C:\path\to\project:/workspace" `
  -v "$env:USERPROFILE\.config\earthengine:/root/.config/earthengine" `
  -w /workspace `
  tensorflow/tensorflow:2.14.0-gpu bash


# 2– Install extras that the base image lacks
(root)$ pip install -r requirements.txt
```

### 3.2 Local Setup (optional)

```bash
$ python -m venv .venv && source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

A modern NVIDIA GPU with matching CUDA/cuDNN is strongly advised; CPU‑only training is **slow**.

---

## 4.Dataset Preparation

1. **Google Earth Engine** at https://code.earthengine.google.com/ → create a new project → run `extract_earth_engine.js`, export `big_rgb.tif` & `big_dem.tif`. This may take time, however they could not be included in the .zip as they are about 16GB in size combined. May want to export smaller files just to test.
2. **Tiling**

   ```bash
   $ python grid_data.py
   $ python validate_grid_data.py
   ```
   Results had to be omitted due to size as they are over 10GB.

3. **Pre‑processing & Augmentation**

   ```bash
   $ python pre-process.py           # saves 256×256 RGB+DEM .npy files
   ```

---

## 5.Training the DCGAN

```bash
# Adjust paths & hyper‑params inside gan.py if needed
$ python gan.py                      # trains & writes models to saved_models/
```


---

## 6.Generating New Terrain (Can be run without importing anything, as trained model is included)

### 6.1 Single 256² Blocks

```bash
$ python create_terrain.py \
    --model-path saved_models/generator_model.h5 \
    --num-terrains 8
```

### 6.2 Large Seamless Map (e.g., 4×4 tiles ≈4k×4k)

```bash
$ python create_terrain.py \
    --model-path saved_models/generator_model.h5 \
    --big-grid 4x4 \
    --target-final-size 4097
```

Outputs are 8‑bit RGB (`*_rgb.png`) and 16‑bit heightmaps (`*_height.png`). Statistics are logged to `terrain_generation_stats.csv`.

---



## 7. Troubleshooting

| Symptom                          | Hint                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------- |
| **`cudart64_118.dll not found`** | Ensure host NVIDIA drivers ≥ R535 and `--gpus all` flag passed to Docker        |
| **Tiles look blank/white**       | Verify percentile stretch in `pre‑process.py`; wrong band order often the cause |
| **Visible seams in large map**   | Increase `--edge-match-tries` or `--overlap`; values of 5 / 32 px usually work  |
