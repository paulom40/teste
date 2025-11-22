import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import io
import os

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
    
    # Try to find game name
    if 'Nice' in all_text and 'Marseille' in all_text:
        game_names.append('Nice vs Marseille')
    else:
        # Try to extract game name from title or headers
        title = soup.find('title')
        if title:
            game_names.append(title.get_text(strip=True))
        else:
            h1 = soup.find('h1')
            if h1:
                game_names.append(h1.get_text(strip=True))
            else:
                game_names.append('Game Name Not Found')
    
    # Method 2: Look for prediction data
    # Common patterns for predictions
    prediction_keywords = ['prediction', 'forecast', 'expected', 'predicted', 'tip']
    shot_keywords = ['shots on target', 'sot', 'shots', 'on target']
    corner_keywords = ['corners', 'corner']
    
    # Search for text containing these keywords
    lines = all_text.split('\n')
    for line in lines:
        line_lower = line.lower()
        # Result prediction
        if any(keyword in line_lower for keyword in prediction_keywords):
            if ':' in line or '-' in line:
                result_predictions.append(line.strip())
        # Shots on target
        if any(keyword in line_lower for keyword in shot_keywords):
            if ':' in line or '-' in line:
                shots_on_target.append(line.strip())
        # Corners
        if any(keyword in line_lower for keyword in corner_keywords):
            if ':' in line or '-' in line:
                corners.append(line.strip())
    
    # Method 3: Look for tables
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                header = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                
                if any(keyword in header for keyword in prediction_keywords):
                    result_predictions.append(value)
                elif any(keyword in header for keyword in shot_keywords):
                    shots_on_target.append(value)
                elif any(keyword in header for keyword in corner_keywords):
                    corners.append(value)
    
    # Create DataFrame with proper fallbacks
    data = {
        'Game Name': game_names[:1] if game_names else ['Nice vs Marseille'],
        'Result Prediction': result_predictions[:1] if result_predictions else ['Data not found'],
        'Shots on Target': shots_on_target[:1] if shots_on_target else ['Data not found'],
        'Corners': corners[:1] if corners else ['Data not found']
    }
    
    return pd.DataFrame(data)

def process_excel_data(uploaded_excel, new_data_df):
    """Process the Excel file and add new data to the next empty row in Bets sheet"""
    try:
        # Read the uploaded Excel file
        excel_data = uploaded_excel.read()
        excel_buffer = io.BytesIO(excel_data)
        
        # Read the Bets sheet
        existing_df = pd.read_excel(excel_buffer, sheet_name='Bets')
        
        # Reset buffer for writing
        excel_buffer.seek(0)
        
        st.write(f"Current Excel file has {len(existing_df)} rows in 'Bets' sheet")
        
        # Find the next empty row (checking rows 1-7)
        next_empty_row = None
        for i in range(min(7, len(existing_df))):
            # Check if the row is empty (first cell is NaN or empty)
            if pd.isna(existing_df.iloc[i, 0]) or existing_df.iloc[i, 0] == '':
                next_empty_row = i
                break
        
        if next_empty_row is None:
            # If no empty rows found in 1-7, append at the end
            if len(existing_df) < 7:
                next_empty_row = len(existing_df)
            else:
                next_empty_row = len(existing_df)
                st.warning("All rows 1-7 are filled. Adding data to the next available row.")
        
        st.write(f"Next empty row: {next_empty_row + 1}")
        
        # Add the new data
        if next_empty_row < len(existing_df):
            # Update existing empty row
            for col_idx, col_name in enumerate(new_data_df.columns):
                if col_idx < len(existing_df.columns):
                    existing_df.iloc[next_empty_row, col_idx] = new_data_df.iloc[0, col_idx]
                else:
                    # If new column doesn't exist in original, add it
                    existing_df[col_name] = ''
                    existing_df.iloc[next_empty_row, len(existing_df.columns)-1] = new_data_df.iloc[0, col_idx]
        else:
            # Append new row
            existing_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        
        # Create output buffer
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            existing_df.to_excel(writer, sheet_name='Bets', index=False)
        
        output_buffer.seek(0)
        return output_buffer, next_empty_row + 1
        
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None, None

# Streamlit app
st.title("📊 Betano Soccer Data Extractor")

st.write("""
This tool extracts soccer prediction data from HTML reports and adds it to your Betano Excel file.

**Columns:**
- A: Game Name
- B: Result Prediction  
- C: Shots on Target
- D: Corners

The script will find the next empty row in your Bets sheet (checking rows 1-7 first).
""")

# File upload section
st.subheader("📁 Upload Files")

col1, col2 = st.columns(2)

with col1:
    uploaded_html = st.file_uploader("HTML Prediction File", type=['html'], help="Upload your Nice_vs_Marseille_prediction_report.html file")

with col2:
    uploaded_excel = st.file_uploader("Betano Excel File", type=['xlsx'], help="Upload your Betano.xlsx file")

if uploaded_html and uploaded_excel:
    try:
        # Extract data from HTML
        html_content = uploaded_html.read().decode('utf-8')
        new_data_df = extract_soccer_data(html_content)
        
        st.subheader("✅ Extracted Data")
        st.dataframe(new_data_df)
        
        # Process Excel file
        output_buffer, target_row = process_excel_data(uploaded_excel, new_data_df)
        
        if output_buffer:
            st.subheader("📥 Download Updated File")
            
            st.success(f"Data will be added to row {target_row} in the 'Bets' sheet")
            
            # Download button
            st.download_button(
                label="💾 Download Updated Betano.xlsx",
                data=output_buffer,
                file_name="Betano_updated.xlsx",
                mime="application/vnd.ms-excel",
                help="Click to download the updated Excel file with your new data"
            )
            
            # Show current sheet status
            st.subheader("📊 Current Sheet Status")
            excel_data = uploaded_excel.read()
            status_buffer = io.BytesIO(excel_data)
            status_df = pd.read_excel(status_buffer, sheet_name='Bets')
            
            row_status = []
            for i in range(min(10, len(status_df))):
                row_data = status_df.iloc[i] if i < len(status_df) else pd.Series()
                has_data = False
                if len(row_data) > 0 and not pd.isna(row_data.iloc[0]) and row_data.iloc[0] != '':
                    has_data = True
                
                status = "✅ Has Data" if has_data else "⬜ Empty"
                highlight = "**→ NEW DATA**" if i + 1 == target_row else ""
                row_status.append({
                    "Row": i + 1, 
                    "Status": status,
                    "Action": highlight
                })
            
            status_df_display = pd.DataFrame(row_status)
            st.table(status_df_display)
    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.info("Please make sure:")
        st.info("1. Your Excel file is named 'Betano.xlsx' and has a sheet called 'Bets'")
        st.info("2. The HTML file contains the prediction data")
        st.info("3. Both files are not corrupted")

elif uploaded_html and not uploaded_excel:
    st.warning("⚠️ Please upload your Betano Excel file to continue")

elif not uploaded_html and uploaded_excel:
    st.warning("⚠️ Please upload your HTML prediction file to continue")

# Debug section
with st.expander("🔍 Debug Information"):
    if uploaded_html:
        st.write("HTML file uploaded successfully")
        html_content = uploaded_html.read().decode('utf-8')
        st.write(f"HTML size: {len(html_content)} characters")
        st.text_area("First 500 characters of HTML:", html_content[:500], height=150)
    
    if uploaded_excel:
        st.write("Excel file uploaded successfully")
        try:
            excel_data = uploaded_excel.read()
            excel_buffer = io.BytesIO(excel_data)
            sheets = pd.ExcelFile(excel_buffer).sheet_names
            st.write(f"Sheet names in Excel file: {sheets}")
        except Exception as e:
            st.write(f"Error reading Excel file: {e}")
