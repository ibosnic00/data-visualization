import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import hsv_to_rgb
from enum import Enum

class RentStatType(Enum):
    AVERAGE = "Average"
    MINIMUM = "Minimum"
    MEDIAN = "Median"

class OutputType(Enum):
    GRAPH = "Graph"
    CSV = "csv"

# Function to parse living area (extract numeric part)
def parse_living_area(area_str):
    return float(area_str.split(' ')[0])

# Function to parse location (split city and neighborhood)
def parse_location(location):
    if ',' in location:
        city, neighborhood = location.split(',', 1)
        return city.strip(), neighborhood.strip()
    return '', location.strip()

# Function to filter listings by living area ranges
def filter_by_living_area(data, min_rent_amount, max_rent_amount):
    ranges = {
        '1 - 35 m2': [],
        '36 - 50 m2': [],
        '51 - 65 m2': [],
        '66 - 80 m2': [],
        '81 - 100 m2': [],
        '101+ m2': [],
    }

    for entry in data:
        # Skip entries with rent outside the range
        if entry["price"] > max_rent_amount or entry["price"] < min_rent_amount:
            continue
            
        living_area = parse_living_area(entry["Living Area"])
        city, neighborhood = parse_location(entry["location"])
        price = entry["price"]
        
        # Store both city and neighborhood
        if living_area <= 35:
            ranges['1 - 35 m2'].append((city, neighborhood, price))
        elif living_area <= 50:
            ranges['36 - 50 m2'].append((city, neighborhood, price))
        elif living_area <= 65:
            ranges['51 - 65 m2'].append((city, neighborhood, price))
        elif living_area <= 80:
            ranges['66 - 80 m2'].append((city, neighborhood, price))
        elif living_area <= 100:
            ranges['81 - 100 m2'].append((city, neighborhood, price))
        else:
            ranges['101+ m2'].append((city, neighborhood, price))
    
    return ranges

# Function to calculate the average rent for each location in each living area range
def calculate_average_rent(ranges):
    average_rents = {}

    for range_key, entries in ranges.items():
        location_rents = {}
        location_counts = {}  # Add counter for each location
        city_name = ''
        
        for city, neighborhood, price in entries:
            city_or_neighborhood = city 
            if not city_name and city:  # Store the city name for the title
                city_name = city
            if city_or_neighborhood not in location_rents:
                location_rents[city_or_neighborhood] = []
                location_counts[city_or_neighborhood] = 0
            location_rents[city_or_neighborhood].append(price)
            location_counts[city_or_neighborhood] += 1
        
        # Calculate average rents and store with city name and counts
        average_rents[range_key] = {
            'city': city_name,
            'rents': {location: np.mean(prices) for location, prices in location_rents.items()},
            'counts': location_counts
        }
    
    return average_rents

# Function to load the JSON data from the file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to filter out outlier neighborhoods based on rent prices using the IQR method
def filter_outliers(rents, threshold_multiplier=1.5):
    """
    Filter out outlier neighborhoods based on rent prices using the IQR method.
    
    Args:
        rents (dict): Dictionary with neighborhood names as keys and rent prices as values
        threshold_multiplier (float): IQR multiplier to determine outlier threshold (default: 1.5)
    
    Returns:
        dict: Filtered dictionary with outliers removed
    """
    if not rents or len(rents) < 4:  # Need at least 4 data points for meaningful quartiles
        return rents
    
    # Extract prices and calculate quartiles
    prices = np.array(list(rents.values()))
    q1 = np.percentile(prices, 25)
    q3 = np.percentile(prices, 75)
    iqr = q3 - q1
    
    # Calculate lower and upper bounds
    lower_bound = q1 - (threshold_multiplier * iqr)
    upper_bound = q3 + (threshold_multiplier * iqr)
    
    # Create filtered dictionary excluding outliers
    filtered_rents = {
        location: price 
        for location, price in rents.items() 
        if lower_bound <= price <= upper_bound
    }
    
    # If we filtered too many values (more than 50%), fall back to a percentile-based approach
    if len(filtered_rents) < len(rents) * 0.5:
        p5 = np.percentile(prices, 5)
        p95 = np.percentile(prices, 95)
        filtered_rents = {
            location: price 
            for location, price in rents.items() 
            if p5 <= price <= p95
        }
    
    return filtered_rents

# Function to generate distinct colors
def generate_colors(n, mono=False, base_color='blue'):
    if mono:
        # Dictionary of predefined base colors
        base_colors = {
            'blue': (0.6, 0.8, 0.9),    # HSV values for blue
            'green': (0.35, 0.8, 0.9),  # HSV values for green
            'red': (0.0, 0.8, 0.9),     # HSV values for red
            'purple': (0.8, 0.8, 0.9),  # HSV values for purple
            'orange': (0.1, 0.8, 0.9)   # HSV values for orange
        }
        
        # Get the HSV values for the chosen base color
        h, s, v = base_colors.get(base_color.lower(), base_colors['blue'])
        
        colors = []
        # Vary the saturation and value while keeping the same hue
        for i in range(n):
            # Calculate saturation and value variations
            s_var = 0.4 + (0.5 * i/n)  # Saturation varies from 0.4 to 0.9
            v_var = 0.5 + (0.4 * i/n)  # Value varies from 0.5 to 0.9
            colors.append(hsv_to_rgb((h, s_var, v_var)))
        
        # Shuffle to avoid gradient effect
        np.random.seed(42)
        np.random.shuffle(colors)
        return colors
    else:
        # Original rainbow color generation
        golden_ratio = 0.618033988749895
        colors = []
        h = 0.1  # Start hue
        
        for i in range(n):
            h = (h + golden_ratio) % 1
            colors.append(hsv_to_rgb((h, 0.8, 0.9)))
        
        # Shuffle colors to ensure similar hues are not adjacent
        np.random.seed(42)
        np.random.shuffle(colors)
        return colors

# Modify the calculate_average_and_min_rent function to include median
def calculate_rent_statistics(ranges):
    rent_stats = {}

    for range_key, entries in ranges.items():
        location_rents = {}
        location_counts = {}
        city_name = ''
        
        for city, neighborhood, price in entries:
            city_or_neighborhood = city 
            if not city_name and city:
                city_name = city
            if city_or_neighborhood not in location_rents:
                location_rents[city_or_neighborhood] = []
                location_counts[city_or_neighborhood] = 0
            location_rents[city_or_neighborhood].append(price)
            location_counts[city_or_neighborhood] += 1
        
        # Calculate average, minimum and median rents
        rent_stats[range_key] = {
            'city': city_name,
            'avg_rents': {location: np.mean(prices) for location, prices in location_rents.items()},
            'min_rents': {location: min(prices) for location, prices in location_rents.items()},
            'median_rents': {location: np.median(prices) for location, prices in location_rents.items()},
            'counts': location_counts
        }
    
    return rent_stats

# Modify calculate_overall_statistics to include median
def calculate_overall_statistics(data, min_rent_amount, max_rent_amount):
    location_data = {}
    city_name = ''
    
    for entry in data:
        # Skip entries with rent outside the range
        if entry["price"] > max_rent_amount or entry["price"] < min_rent_amount:
            continue
            
        city, neighborhood = parse_location(entry["location"])
        city_or_neighborhood = city
        price = entry["price"]
        living_area = parse_living_area(entry["Living Area"])
        
        if not city_name and city:
            city_name = city
            
        if city_or_neighborhood not in location_data:
            location_data[city_or_neighborhood] = {
                'prices': [],
                'areas': [],
                'count': 0,
                'min_price_area': None
            }
            
        # Update min_price_area if this is the lowest price seen for this neighborhood
        if not location_data[city_or_neighborhood]['prices'] or price < min(location_data[city_or_neighborhood]['prices']):
            location_data[city_or_neighborhood]['min_price_area'] = living_area
            
        location_data[city_or_neighborhood]['prices'].append(price)
        location_data[city_or_neighborhood]['areas'].append(living_area)
        location_data[city_or_neighborhood]['count'] += 1
    
    # Calculate statistics
    stats = {
        'city': city_name,
        'avg_rents': {loc: np.mean(data['prices']) for loc, data in location_data.items()},
        'min_rents': {loc: min(data['prices']) for loc, data in location_data.items()},
        'median_rents': {loc: np.median(data['prices']) for loc, data in location_data.items()},
        'areas': {loc: round(np.mean(data['areas'])) for loc, data in location_data.items()},
        'min_price_areas': {loc: data['min_price_area'] for loc, data in location_data.items()},
        'counts': {loc: data['count'] for loc, data in location_data.items()}
    }
    
    return stats

def is_duplicate_property(prop1, prop2, price_threshold=0.1, area_threshold=0.1):
    """
    Check if two properties are likely duplicates based on price, location, and living area.
    
    Args:
        prop1 (dict): First property dictionary
        prop2 (dict): Second property dictionary
        price_threshold (float): Maximum allowed price difference (default: 0.1 or 10%)
        area_threshold (float): Maximum allowed area difference (default: 0.1 or 10%)
    
    Returns:
        bool: True if properties are likely duplicates, False otherwise
    """
    # Compare locations
    loc1_city, loc1_neighborhood = parse_location(prop1["location"])
    loc2_city, loc2_neighborhood = parse_location(prop2["location"])
    
    if loc1_neighborhood != loc2_neighborhood:
        return False
    
    # Compare prices within threshold
    price1, price2 = prop1["price"], prop2["price"]
    price_diff = abs(price1 - price2) / max(price1, price2)
    
    if price_diff > price_threshold:
        return False
    
    # Compare living areas within threshold
    area1 = parse_living_area(prop1["Living Area"])
    area2 = parse_living_area(prop2["Living Area"])
    area_diff = abs(area1 - area2) / max(area1, area2)
    
    return area_diff <= area_threshold

def find_duplicate_properties(data, price_threshold=0.1, area_threshold=0.1):
    """
    Find all duplicate properties in the dataset.
    
    Args:
        data (list): List of property dictionaries
        price_threshold (float): Maximum allowed price difference (default: 0.1 or 10%)
        area_threshold (float): Maximum allowed area difference (default: 0.1 or 10%)
    
    Returns:
        list: List of tuples containing indices of duplicate properties
    """
    duplicates = []
    
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if is_duplicate_property(data[i], data[j], price_threshold, area_threshold):
                duplicates.append((i, j))
    
    return duplicates

def remove_duplicates(data, price_threshold=0.01, area_threshold=0.01):
    """
    Remove duplicate properties from the dataset, keeping the first occurrence.
    
    Args:
        data (list): List of property dictionaries
        price_threshold (float): Maximum allowed price difference (default: 0.05 or 5%)
        area_threshold (float): Maximum allowed area difference (default: 0.01 or 1%)
    
    Returns:
        list: List of property dictionaries with duplicates removed
    """
    duplicates = find_duplicate_properties(data, price_threshold, area_threshold)
    duplicate_indices = set(j for _, j in duplicates)  # Keep first occurrence, remove subsequent ones
    
    # Print information about removed duplicates
    if duplicates:
        if verbose:
            print(f"\nFound {len(duplicates)} duplicate properties:")
        for i, j in duplicates:
            prop1, prop2 = data[i], data[j]
            if verbose:
                print(f"Duplicate: {prop2['location']} - {prop2['Living Area']} - €{prop2['price']}")
                print(f"Link: https://www.njuskalo.hr{prop2.get('link', 'No link available')}")
                print(f"Original: {prop1['location']} - {prop1['Living Area']} - €{prop1['price']}")
                print(f"Link: https://www.njuskalo.hr{prop1.get('link', 'No link available')}\n")
    else:
        if verbose:
            print("\nNo duplicate properties found.")
    
    deduplicated = [prop for i, prop in enumerate(data) if i not in duplicate_indices]
    if verbose:
        print(f"Removed {len(data) - len(deduplicated)} duplicate properties.")
        print(f"Dataset size reduced from {len(data)} to {len(deduplicated)} properties.\n")
    
    return deduplicated

def prepare_visualization_data(file_name, data_date, min_rent_amount, max_rent_amount, stat_type=RentStatType.AVERAGE):
    """
    Prepare data for visualization by loading, deduplicating and calculating statistics.
    
    Args:
        file_name (str): Name of the JSON file to load
        min_rent_amount (float): Minimum rent amount to consider
        max_rent_amount (float): Maximum rent amount to consider
        stat_type (RentStatType): Type of statistic to calculate (AVERAGE, MINIMUM, MEDIAN)
    
    Returns:
        tuple: (overall_data, rent_stats, rent_type, rent_type_label)
    """
    # Get the absolute path to the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(current_dir, 'data', data_date, file_name + '.json')

    # Load and deduplicate the data
    data = load_data(file_path)
    data = remove_duplicates(data)
    
    # Generate overall statistics
    overall_data = calculate_overall_statistics(data, min_rent_amount, max_rent_amount)
    
    # Get rent type mapping
    rent_type_map = {
        RentStatType.AVERAGE: 'avg_rents',
        RentStatType.MINIMUM: 'min_rents',
        RentStatType.MEDIAN: 'median_rents'
    }
    rent_type = rent_type_map[stat_type]
    rent_type_label = stat_type.value

    # Prepare size-specific data
    ranges = filter_by_living_area(data, min_rent_amount, max_rent_amount)
    rent_stats = calculate_rent_statistics(ranges)

    return overall_data, rent_stats, rent_type, rent_type_label

def generate_graphs(overall_data, rent_stats, rent_type, rent_type_label, show_all_sizes_in_one_graph=False):
    """
    Display rent statistics graphs.
    """

    if show_all_sizes_in_one_graph:
        # Create overall statistics graph only
        plt.figure(figsize=(10, 6))
        
        # Filter outliers from overall data only for average prices
        if stat_type == RentStatType.AVERAGE:
            filtered_overall = filter_outliers(overall_data[rent_type])
        else:
            filtered_overall = overall_data[rent_type]
        
        # Sort locations by price in descending order
        sorted_items = sorted(filtered_overall.items(), key=lambda x: x[1], reverse=True)
        locations = [loc for loc, _ in sorted_items]
        prices = [price for _, price in sorted_items]
        
        # Generate distinct colors (monochrome blue)
        colors = generate_colors(len(locations), mono=True, base_color='blue')
        
        # Create bars with different colors
        bars = plt.bar(range(len(locations)), prices, color=colors)
        
        # Add value labels above each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'€{int(height)}',
                    ha='center', va='bottom')
        
        # Create colored labels
        plt.xticks(range(len(locations)), 
                  ['' for _ in locations],
                  rotation=45)
        
        # Add colored text labels manually
        ax = plt.gca()
        for idx, (label, color) in enumerate(zip(locations, colors)):
            text = f"{label} [{overall_data['counts'][label]}] {{{overall_data['areas'][label]}m²}}"
            ax.text(idx, -0.1, text,
                   rotation=45,
                   ha='right',
                   va='top',
                   color=color,
                   transform=ax.get_xaxis_transform(),
                   fontweight='bold')
        
        plt.title(f"{rent_type_label} Rent (All Sizes)")
        plt.ylabel(f"{rent_type_label} Rent (€)")
        plt.tight_layout()
        
    else:
        # Create separate windows for each range
        for range_key, data in rent_stats.items():
            city = data['city']
            rents = data[rent_type]
            counts = data['counts']
            
            # Filter out outliers only for average prices
            if stat_type == RentStatType.AVERAGE:
                filtered_rents = filter_outliers(rents)
            else:
                filtered_rents = rents
            
            # Create a new figure for each range
            plt.figure(figsize=(10, 6))
            
            # Sort locations by price in descending order
            sorted_items = sorted(filtered_rents.items(), key=lambda x: x[1], reverse=True)
            locations = [loc for loc, _ in sorted_items]
            prices = [price for _, price in sorted_items]
            
            # Generate distinct colors (monochrome blue)
            colors = generate_colors(len(locations), mono=True, base_color='blue')
            
            # Create bars with different colors
            bars = plt.bar(range(len(locations)), prices, color=colors)
            
            # Add value labels above each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'€{int(height)}',
                        ha='center', va='bottom')
            
            # Create colored labels
            plt.xticks(range(len(locations)), 
                      ['' for _ in locations],
                      rotation=45)
            
            # Add colored text labels manually
            ax = plt.gca()
            for idx, (label, color) in enumerate(zip(locations, colors)):
                text = f"{label} [{counts[label]}]"
                ax.text(idx, -0.20, text,
                       rotation=45,
                       ha='right',
                       va='top',
                       color=color,
                       transform=ax.get_xaxis_transform(),
                       fontweight='bold')
            
            plt.title(f"{rent_type_label} Rent for {range_key}")
            plt.ylabel(f"{rent_type_label} Rent (€)")
            
            plt.subplots_adjust(bottom=0.01)
            plt.tight_layout()

    plt.show()

def generate_csv(overall_data, rent_stats, rent_type, rent_type_label, show_all_sizes_in_one_graph=False):
    """
    Generate CSV output of rent statistics and write to file.
    """
    # Create output filename based on input JSON
    output_file = f"generated_data_{file_name}.csv"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if show_all_sizes_in_one_graph:
            # Generate overall statistics CSV
            f.write(f"{rent_type_label} Rent (All Sizes)\n")
            f.write("Neighborhood,Rent,Sample Size,Average Area\n")
            
            # Sort locations by rent in descending order
            sorted_items = sorted(overall_data[rent_type].items(), key=lambda x: x[1], reverse=True)
            
            for location, rent in sorted_items:
                count = overall_data['counts'][location]
                area = overall_data['areas'][location]
                f.write(f"{location},€{int(rent)},{count},{area}m²\n")
        else:
            # Generate size-specific CSVs
            for range_key, data in rent_stats.items():
                city = data['city']
                rents = data[rent_type]
                counts = data['counts']
                
                f.write(f"\n{rent_type_label} Rent for {range_key}\n")
                f.write("Neighborhood,Rent,Sample Size\n")
                
                # Sort locations by rent in descending order
                sorted_items = sorted(rents.items(), key=lambda x: x[1], reverse=True)
                
                for location, rent in sorted_items:
                    count = counts[location]
                    f.write(f"{location},€{int(rent)},{count}\n")
    
    print(f"Data has been written to {output_file}")

# Settings:
verbose = False
min_rent_amount = 200
max_rent_amount = 3000
output_type = OutputType.CSV
show_overall_statistics = False
file_name = 'iznajmljivanje-stanova-zagreb'
data_date = '05-02-2025'
stat_type = RentStatType.AVERAGE
# city_or_neighborhood = city/neighborhood

# Get prepared data
overall_data, rent_stats, rent_type, rent_type_label = prepare_visualization_data(
    file_name, data_date, min_rent_amount, max_rent_amount, stat_type
)

# Generate output based on type
if output_type == OutputType.GRAPH:
    generate_graphs(overall_data, rent_stats, rent_type, rent_type_label, show_overall_statistics)
else:  # OutputType.CSV
    generate_csv(overall_data, rent_stats, rent_type, rent_type_label, show_overall_statistics)  

