import pandas as pd
from bs4 import BeautifulSoup
import os

def extract_soccer_data(html_file_path, excel_output_path):
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize lists for each column
    game_names = []
    result_predictions = []
    shots_on_target = []
    corners = []
    
    # Extract data - you'll need to adjust these selectors based on your HTML structure
    # These are common CSS classes/selectors used in sports prediction reports
    
    # Example selectors (adjust based on your actual HTML structure):
    
    # 1. Game name (look for team names, match title, etc.)
    game_elements = soup.find_all('div', class_=['match-title', 'team-names', 'fixture'])
    for element in game_elements[:1]:  # Assuming one main game
        game_names.append(element.get_text(strip=True))
    
    # 2. Result prediction (look for prediction, score forecast, etc.)
    prediction_elements = soup.find_all('div', class_=['prediction', 'forecast', 'expected-score'])
    for element in prediction_elements[:1]:
        result_predictions.append(element.get_text(strip=True))
    
    # 3. Shots on Target (look for statistics, shots data)
    shots_elements = soup.find_all('span', class_=['shots-on-target', 'sot', 'stat-value'])
    for element in shots_elements[:2]:  # Assuming data for both teams
        shots_on_target.append(element.get_text(strip=True))
    
    # 4. Corners (look for corners statistics)
    corners_elements = soup.find_all('span', class_=['corners', 'corner-stats', 'stat-value'])
    for element in corners_elements[:2]:  # Assuming data for both teams
        corners.append(element.get_text(strip=True))
    
    # If the above selectors don't work, try these alternative approaches:
    
    # Alternative: Look for tables containing the data
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4:
                # Adjust indices based on your table structure
                game_names.append(cells[0].get_text(strip=True))
                result_predictions.append(cells[1].get_text(strip=True))
                shots_on_target.append(cells[2].get_text(strip=True))
                corners.append(cells[3].get_text(strip=True))
    
    # If still no data, print the HTML structure to inspect
    if not game_names:
        print("No data found. Here's the HTML structure to help identify the right selectors:")
        print(soup.prettify()[:2000])  # First 2000 characters
    
    # Create DataFrame
    data = {
        'Game Name': game_names if game_names else ['Nice vs Marseille'],
        'Result Prediction': result_predictions if result_predictions else ['No prediction found'],
        'Shots on Target': [' vs '.join(shots_on_target)] if shots_on_target else ['No data'],
        'Corners': [' vs '.join(corners)] if corners else ['No data']
    }
    
    df = pd.DataFrame(data)
    
    # Export to Excel
    try:
        # Try to read existing file first
        try:
            existing_df = pd.read_excel(excel_output_path, sheet_name='Bets')
            # Append new data
            final_df = pd.concat([existing_df, df], ignore_index=True)
        except:
            # File doesn't exist or sheet doesn't exist, create new
            final_df = df
        
        # Save to Excel
        with pd.ExcelWriter(excel_output_path, engine='openpyxl', mode='w') as writer:
            final_df.to_excel(writer, sheet_name='Bets', index=False)
        
        print(f"Data successfully exported to {excel_output_path}")
        print(f"Added {len(df)} row(s) to the 'Bets' sheet")
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")

# Usage
if __name__ == "__main__":
    # Update these paths with your actual file paths
    html_file_path = r"C:\Users\paulo\OneDrive\Desktop\SOCCER\Nice_vs_Marseille_prediction_report.html"
    excel_output_path = r"C:\Users\paulo\OneDrive\Desktop\Bets.xlsx"  # Update with your Excel file path
    
    extract_soccer_data(html_file_path, excel_output_path)
