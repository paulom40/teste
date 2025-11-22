import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import io
import requests

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

def get_next_empty_row(excel_buffer=None):
    """Check which rows 1-7 have data and return the next empty row"""
    try:
        if excel_buffer:
            # Read existing Excel file from buffer
            existing_df = pd.read_excel(excel_buffer, sheet_name='Bets')
        else:
            st.info("No existing Excel file provided. Starting from row 1.")
            return 1
    except:
        st.info("No existing 'Bets' sheet found. Starting from row 1.")
        return 1
    
    # Check rows 1-7 for data
    filled_rows = []
    for i in range(min(7, len(existing_df))):
        row_data = existing_df.iloc[i] if i < len(existing_df) else None
        # Check if row has any non-null data in the first 4 columns
        if row_data is not None and not pd.isna(row_data).all():
            if len(row_data) >= 4 and not pd.isna(row_data.iloc[0]):
                filled_rows.append(i + 1)  # +1 because Excel rows start at 1
    
    st.write(f"Rows with data: {filled_rows}")
    
    if filled_rows:
        next_empty_row = max(filled_rows) + 1
        # Make sure we don't go beyond row 7 for new data
        if next_empty_row > 7:
            st.warning("All rows 1-7 are filled. New data will be added starting from row 8.")
        return next_empty_row
    else:
        return 1

# Streamlit app
st.title("Soccer Prediction Data Extractor")

st.write("""
Upload your HTML prediction report file and I'll extract the data for you.
The data will be organized in columns:
- Column A: Game Name
- Column B: Result Prediction  
- Column C: Shots on Target
- Column D: Corners

The script will check rows 1-7 in your existing Bets sheet and add new data to the next empty row.
""")

# File uploaders
st.subheader("Step 1: Upload Files")

col1, col2 = st.columns(2)

with col1:
    uploaded_html = st.file_uploader("Upload HTML Prediction File", type=['html'], key="html")

with col2:
    uploaded_excel = st.file_uploader("Upload Existing Excel File (Optional)", type=['xlsx'], key="excel")

if uploaded_html is not None:
    # Read the uploaded HTML file
    html_content = uploaded_html.read().decode('utf-8')
    
    # Extract data from HTML
    new_data_df = extract_soccer_data(html_content)
    
    st.subheader("Step 2: Extracted Data Preview")
    st.dataframe(new_data_df)
    
    # Show raw HTML structure for debugging
    with st.expander("Debug: Show HTML Structure (first 2000 chars)"):
        st.text(html_content[:2000])
    
    st.subheader("Step 3: Export to Excel")
    
    if uploaded_excel is not None:
        # Read existing Excel file
        excel_buffer = io.BytesIO(uploaded_excel.read())
        
        # Find next empty row
        next_row = get_next_empty_row(excel_buffer)
        st.write(f"Next empty row for new data: **Row {next_row}**")
        
        try:
            # Read existing data
            existing_df = pd.read_excel(excel_buffer, sheet_name='Bets')
            
            # Ensure existing_df has at least next_row-1 rows
            while len(existing_df) < next_row - 1:
                existing_df = pd.concat([existing_df, pd.DataFrame([{}] * (next_row - 1 - len(existing_df)))], ignore_index=True)
            
            # Add new data at the correct row position
            if next_row - 1 < len(existing_df):
                # Update existing row
                for col_idx, col_name in enumerate(new_data_df.columns):
                    if col_idx < len(existing_df.columns):
                        existing_df.iloc[next_row - 1, col_idx] = new_data_df.iloc[0, col_idx]
            else:
                # Add new row
                existing_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            
            # Create download buffer
            output_buffer = io.BytesIO()
            existing_df.to_excel(output_buffer, index=False, sheet_name='Bets')
            output_buffer.seek(0)
            
            # Download button
            st.download_button(
                label=f"Download Updated Excel File (Data added to row {next_row})",
                data=output_buffer,
                file_name="updated_soccer_predictions.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
    
    else:
        # Create new Excel file
        st.info("No existing Excel file uploaded. Creating a new one.")
        
        output_buffer = io.BytesIO()
        new_data_df.to_excel(output_buffer, index=False, sheet_name='Bets')
        output_buffer.seek(0)
        
        st.download_button(
            label="Download New Excel File",
            data=output_buffer,
            file_name="soccer_predictions.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    st.success("Data extraction complete!")
    
    # Show current row status
    st.subheader("Current Row Status")
    if uploaded_excel is not None:
        try:
            excel_buffer.seek(0)
            status_df = pd.read_excel(excel_buffer, sheet_name='Bets')
            row_status = []
            for i in range(min(10, len(status_df))):  # Check first 10 rows
                row_data = status_df.iloc[i] if i < len(status_df) else None
                if row_data is not None and len(row_data) >= 1 and not pd.isna(row_data.iloc[0]):
                    status = "✅ Has Data"
                else:
                    status = "⬜ Empty"
                row_status.append({"Row": i + 1, "Status": status})
            
            st.table(pd.DataFrame(row_status))
        except:
            st.info("Could not read row status from uploaded Excel file.")
