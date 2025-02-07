import json
import os
from collections import defaultdict

def load_neighbourhood_config():
    with open('neighbourhood_configuration.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_neighbourhood(location, neighbourhoods_dict):
    # Extract location before comma if exists
    area = location.split(',')[1].strip() if ',' in location else location.strip()
    
    # Check which neighborhood contains this area
    for neighbourhood, areas in neighbourhoods_dict.items():
        if any(area in location for area in areas):
            return neighbourhood
            
    return "Other"  # Default if no match found

def get_zagreb_neighbourhood(location):
    return location.split(',')[0].strip()

def modify_json_files():
    other_neighbourhoods = defaultdict(set)  # To store locations categorized as "Other"
    neighbourhood_config = load_neighbourhood_config()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each JSON file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            input_filepath = os.path.join(data_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            
            # Extract city name correctly by joining all parts after "iznajmljivanje-stanova-"
            city = '-'.join(filename.split('-')[2:]).replace('.json', '').lower()
            
            # Read JSON file
            with open(input_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if not a list or empty
            if not isinstance(data, list) or not data:
                continue
            
            # Add neighbourhood and city fields to each entry
            for entry in data:
                if 'location' in entry:
                    # Set city field with proper capitalization
                    city_names = {
                        'split': 'Split',
                        'rijeka': 'Rijeka',
                        'zadar': 'Zadar',
                        'pula': 'Pula',
                        'solin': 'Solin',
                        'velika-gorica': 'Velika Gorica',
                        'slavonski-brod': 'Slavonski Brod',
                        'osijek': 'Osijek',
                        'varazdin': 'Varaždin',
                        'karlovac': 'Karlovac',
                        'zagreb': 'Zagreb'
                    }
                    entry['city'] = city_names.get(city, city.capitalize())
                    
                    # Get neighbourhood
                    if city == 'zagreb':
                        neighbourhood = get_zagreb_neighbourhood(entry['location'])
                    else:
                        neighbourhood = get_neighbourhood(entry['location'], 
                                                       neighbourhood_config.get(city, {}))
                    
                    entry['neighbourhood'] = neighbourhood
                    
                    # Log if neighbourhood is "Other"
                    if neighbourhood == "Other":
                        area = entry['location'].split(',')[1].strip() if ',' in entry['location'] else entry['location'].strip()
                        other_neighbourhoods[city].add(area)
            
            # Write modified data to output directory
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print summary of unmatched neighborhoods
    print("\nLocations categorized as 'Other' by city:")
    for city, locations in other_neighbourhoods.items():
        if locations:  # Only print if there are any "Other" locations
            print(f"\n{city.upper()}:")
            for location in sorted(locations):
                print(f"  - {location}")
    if (len(other_neighbourhoods.items()) == 0):
        print("No 'Other' locations found")


#settings
data_dir = "data/05-02-2025"
output_dir = "data/05-02-2025-modified"

if __name__ == "__main__":
    modify_json_files()
