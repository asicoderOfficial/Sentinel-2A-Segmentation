# ArchitectureOfMLSystemsSoSe24

# AMLS Project

## Development of Machine Learning (ML) Pipelines

### Project Overview

This project involves the development of machine learning pipelines to classify building locations in cities from satellite images. The goal is to train models that can determine whether pixels in satellite images contain buildings or not.

### Description

#### Data Source

We will be using data from the Sentinel Copernicus program, specifically from Sentinel 2 Satellites.

- **Sentinel 2 Satellites**: [Sentinel 2 Mission](https://sentiwiki.copernicus.eu/web/s2-mission)
- **Processing Levels**: [Sentinel 2 MSI Processing Levels](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels)

The baseline solution must utilize the processing level 2a with 10-metre spatial resolution bands as inputs.

#### Test Data

The test data for all accuracy results presented in the report should use the following coordinates:

- **Latitude**:

  - North: 52.574409
  - South: 52.454927

- **Longitude**:
  - West: 13.294333
  - East: 13.500205

### Tasks

#### TASK 1.1 Data Acquisition and Alignment (15/100 points)

Create a pipeline that constructs training data based on latitude-longitude coordinates. The pipeline must accomplish the following tasks:

1. **Download Open Street Maps files** and create map projections of the contained buildings.
2. **Download satellite images** from a Sentinel 2 data provider.

### Structure

- `data_acquisition/`
  - Scripts and tools for downloading and aligning data.
- `models/`
  - Trained models and training scripts.
- `notebooks/`
  - Jupyter notebooks for exploratory data analysis and visualization.
- `reports/`
  - Accuracy results and performance metrics.

### Setup

1. **Clone the repository:**
   ```bash
   git clone .git
   cd amls-project
   ```
