# Rental Market Analysis Tool

A Python-based data analysis tool for visualizing and analyzing rental property data from Njuskalo.hr.

## Overview
This tool processes rental property data to generate statistical insights and visualizations about rental market trends across different neighborhoods and property sizes.

## Features

### Statistical Analysis
- Average rent calculations
- Median rent calculations
- Minimum rent tracking
- Sample size monitoring
- Outlier filtering using IQR method

### Visualization Options
- Bar charts showing rent statistics by neighborhood
- Color-coded visualization with property counts
- Automatic outlier filtering for cleaner visualization
- Value labels showing exact rent amounts
- Option to show all property sizes in one graph or separate graphs

### Data Processing
- Property size categorization (1-35m², 36-50m², etc.)
- Duplicate property detection and removal
- Location-based grouping (city and neighborhood)
- Configurable rent range filtering

## Usage

### Basic Example
```python
from data_parser import *

# Configure analysis parameters
min_rent_amount = 200
max_rent_amount = 3000
output_type = OutputType.GRAPH
stat_type = RentStatType.AVERAGE
json_file_name = 'iznajmljivanje-stanovasolin'

# Generate visualization
overall_data, rent_stats, rent_type, rent_type_label = prepare_visualization_data(
    json_file_name, min_rent_amount, max_rent_amount, stat_type
)
generate_graphs(overall_data, rent_stats, rent_type, rent_type_label, False)
```

### Configuration Options

#### Statistical Types (RentStatType)
- `AVERAGE`: Mean rental price
- `MINIMUM`: Lowest rental price
- `MEDIAN`: Middle value rental price

#### Output Types (OutputType)
- `GRAPH`: Generate matplotlib visualizations
- `CSV`: Export data to CSV format

#### Other Parameters
- `min_rent_amount`: Minimum rent to consider
- `max_rent_amount`: Maximum rent to consider
- `show_all_sizes_in_one_graph`: Toggle between combined or separate size graphs

### Output Formats

#### Graph Output
The tool generates bar charts with:
- Neighborhood-based rent comparisons
- Color-coded bars for easy differentiation
- Sample size indicators
- Property size information
- Exact rent amount labels

#### CSV Output
Generates structured data files containing:
- Neighborhood names
- Rent amounts
- Sample sizes
- Property size ranges
- Statistical calculations

## Dependencies
- NumPy: Statistical computations
- Matplotlib: Data visualization
- JSON: Data input parsing

## Data Processing Details

### Duplicate Detection
Properties are considered duplicates when they share:
- Same neighborhood location
- Similar price (configurable threshold)
- Similar living area (configurable threshold)

### Outlier Filtering
Uses IQR (Interquartile Range) method with:
- Configurable threshold multiplier
- Fallback to percentile-based filtering
- Preservation of minimum sample sizes

### Size Categories
Properties are grouped into ranges:
- 1 - 35 m²
- 36 - 50 m²
- 51 - 65 m²
- 66 - 80 m²
- 81 - 100 m²
- 101 - 120 m²
- 121+ m²

## Input Data Format
Expects JSON files with property entries:
```json
{
  "name": "ADVERT NAME",
  "location": "LOCATION DATA",
  "time": "DATE POSTED",
  "price": "PRICE",
  "Living Area": "AREA IN M2"
}
```

For more information about data collection, see the main project README.
