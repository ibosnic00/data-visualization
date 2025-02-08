# Rental Market Analysis Tool

A Python-based data analysis tool for visualizing and analyzing rental property data from json files.

## Overview
This tool processes rental property data through a two-step pipeline:
1. Data modification (`data_modifier.py`) - Enriches raw data with neighborhood classifications
2. Data analysis and visualization (`data_parser.py`) - Generates statistical insights and visualizations

## Pipeline Components

### 1. Data Modifier (`data_modifier.py`)
- Enriches raw JSON data with additional location context
- Uses `neighbourhood_configuration.json` to map locations to defined districts/areas
- Features:
  - Adds city and neighborhood fields to each property entry
  - Supports custom neighborhood mapping for each city
  - Special handling for Zagreb's district format
  - Logs unmatched locations as "Other" for future configuration updates
  - Preserves original data while adding enriched fields

### 2. Data Parser (`data_parser.py`)
Processes the enriched data to generate insights and visualizations.

#### Statistical Analysis Features
- Multiple statistical measures:
  - Average rent calculations
  - Median rent calculations
  - Minimum rent tracking
- Sample size monitoring
- Outlier filtering using IQR method
- Duplicate property detection and removal

#### Property Categorization
- Size ranges:
  - 1 - 35 m²
  - 36 - 50 m²
  - 51 - 65 m²
  - 66 - 80 m²
  - 81 - 100 m²
  - 101+ m²

#### Output Options
- Visualization Types:
  - Bar charts with rent statistics by neighborhood
  - Color-coded visualization with property counts
  - Value labels showing exact rent amounts
  - Combined or separate size-range graphs
- Data Export:
  - CSV format with detailed statistics
  - Configurable output directory

## Configuration

### Neighborhood Configuration
The `neighbourhood_configuration.json` file defines district mappings for each city:
```json
{
  "city-name": {
    "District Category": [
      "Area1", "Area2", "Area3"
    ]
  }
}
```

### Analysis Parameters
- `min_rent_amount`: Minimum rent to consider (default: 200€)
- `max_rent_amount`: Maximum rent to consider (default: 3000€)
- `stat_type`: Type of statistic to calculate (AVERAGE, MINIMUM, MEDIAN)
- `output_type`: Output format (GRAPH, CSV)
- `neighborhood_grouping_type`: Grouping level (CITY, NEIGHBORHOOD)

## Usage

### Basic Pipeline Example
```python
# Step 1: Modify and enrich data
python data_modifier.py

# Step 2: Generate analysis
python data_parser.py
```

### Customization Options
```python
# In data_parser.py
output_type = OutputType.GRAPH  # or OutputType.CSV
stat_type = RentStatType.AVERAGE  # or MINIMUM, MEDIAN
neighborhood_grouping_type = NeighborhoodGroupingType.NEIGHBORHOOD  # or CITY
show_overall_statistics = False  # True for combined view
```

## Input Data Format
Expects JSON files with property entries:
```json
{
  "name": "Property Name",
  "location": "Location Data",
  "time": "Date Posted",
  "price": "Price in EUR",
  "Living Area": "Area in m²"
}
```

## Output
### CSV Format
- Neighborhood-wise statistics
- Size-range specific data
- Sample sizes and average areas
- Sorted by rent values (descending)

### Graph Format
- Interactive matplotlib visualizations
- Color-coded districts
- Sample size indicators
- Exact rent amount labels
- Outlier filtering for cleaner visualization

## Dependencies
- NumPy: Statistical computations
- Matplotlib: Data visualization
- JSON: Data processing
