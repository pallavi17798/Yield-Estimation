# Yield Estimation

## Overview
The **Yield Estimation** project is designed to estimate crop yields using advanced machine learning techniques. The project utilizes various data sources, including satellite imagery, to analyze and predict crop performance at different growth stages.


## Features
- **Crop Stage Detection**: SAVI, NDVI and NDRE Remote sensing indices have taken into consideration to input stagewise values of crop.
- **Yield Prediction**: Estimates the expected yield based on various input parameters (SAVI, NDVI, NDRE, Min and Max Temp, APAR, Rainfall and land surface temperature) and these meteorological parameters have extracted from NASA POWER Portal.
- **Data Visualization**: Visualizes crop data for better insights and understanding with the help of matplotlib
- **Integration with Google Earth Engine**: Utilizes satellite imagery for accurate crop monitoring and extraction of remote sensing indices.

## Technologies Used
- Python
- TensorFlow
- Google Earth Engine
- NumPy
- Pandas
- SciPy
- Scikit-learn
- STAC API (Element-84)

## Getting Started

### Prerequisites
To run this project, ensure you have the following installed:

- Python 3.x
- Pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pallavi17798/Yield-Estimation.git
   cd Yield-Estimation

