# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:24:25 2025

@author: Usuario
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load Excel file without explicit headers
df = pd.read_excel('ruscorpora_content (1)-вводныеСлова.xlsx', header=None)

# Function to create augmented context and marker
def augment_row(row, marker_idx=3):
    left_context = " ".join(str(row[col]) for col in range(0, marker_idx) if pd.notna(row[col]))
    marker = str(row[marker_column]) if pd.notna(row[marker_column]) else ""
    right_context = " ".join(str(row[col]) for col in range(marker_column+1, len(row)) if pd.notna(row[col]))
    context = f"{left_context} {marker} {right_context}".strip()
    return context, marker

# Define marker column index
marker_column = 3

# Generate augmented dataset
augmented_data = []
for _, row in df.iterrows():
    input_text, marker = augment_row(row)
    # Original data entry
    augmented_data.append({"input": input_text, "output": marker})
    # Simple duplication for augmentation
    augmented_data.append({"input": input_text, "output": marker})

# Split data into training (80%) and testing (20%)
train_data, test_data = train_test_split(augmented_data, test_size=0.2, random_state=42)

# Export training data to JSON
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

# Export testing data to JSON
with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("Augmented data has been successfully split and saved into train_data.json and test_data.json.")
