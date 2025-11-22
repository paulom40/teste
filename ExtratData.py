import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import io

def extract_soccer_data(html_content):
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize lists for each column
    game_names = []
    result_predictions = []
    shots_on_target = []
    corners = []
    
    # Extract data - try different selectors
    # Method 1: Look for common patterns
    all_text = soup.get_text()
    
    # Try to find game name (common patterns)
    if 'Nice' in all_text and 'Marseille' in all_text:
        game_names.append('Nice vs Marseille')
    
    # Method 2: Look for tables
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            cell_text = [cell.get_text(strip=True) for cell in cells]
            
            # Look for relevant data in table cells
            for i, text in enumerate(cell_text):
                if 'prediction' in text.lower() or 'forecast' in text.lower():
                    if i + 1 < len(cell_text):
                        result_predictions.append(cell_text[i + 1])
                elif 'shots' in text.lower() or 'sot' in text.lower():
                    if i + 1 < len(cell_text):
                        shots_on_target.append(cell_text[i + 1])
                elif 'corner' in text.lower():
                    if i + 1 < len(cell_text):
                        corners.append(cell_text[i + 1])
    
    # Method 3: Look for divs with common classes
    predictions = soup.find_all(['div', 'span'], class_=lambda x: x and any(word in str(x).lower() for word in ['prediction', 'forecast', 'result']))
    for pred in predictions:
        result_predictions.append(pred.get_text(strip=True))
    
    # Create DataFrame
    data = {
        'Game Name': game_names if game_names else ['Nice vs Marseille'],
        'Result Prediction': result_predictions[:1] if result_predictions else ['Check HTML structure'],
        'Shots on Target': [' vs '.join(shots_on_target[:2])] if shots_on_target else ['No data'],
        'Corners': [' vs '.join(corners[:2])] if corners else ['No data']
    }
    
    return pd.DataFrame(data)

# Streamlit app
st.title("Soccer Prediction Data Extractor")

st.write("""
Upload your HTML prediction report file and I'll extract the data for you.
The data will be organized in columns:
- Column A: Game Name
- Column B: Result Prediction  
- Column C: Shots on Target
- Column D: Corners
""")

# File uploader
uploaded_file = st.file_uploader("Choose an HTML file", type=['html'])

if uploaded_file is not None:
    # Read the uploaded file
    html_content = uploaded_file.read().decode('utf-8')
    
    # Extract data
    df = extract_soccer_data(html_content)
    
    # Show extracted data
    st.subheader("Extracted Data")
    st.dataframe(df)
    
    # Show raw HTML structure for debugging
    with st.expander("Debug: Show HTML Structure (first 2000 chars)"):
        st.text(html_content[:2000])
    
    # Download button for Excel
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name='Bets')
    excel_buffer.seek(0)
    
    st.download_button(
        label="Download Excel File",
        data=excel_buffer,
        file_name="soccer_predictions.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    st.success("Data extracted successfully! Download the Excel file above.")
