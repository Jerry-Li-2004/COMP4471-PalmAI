import json
import os
import re


def extract_image_number(image_path):
    """Extract the numeric part from image filename like IMG_0182"""
    filename = os.path.basename(image_path)
    # Extract numbers from filename (handles both IMG_0182 and IMG0182 formats)
    match = re.search(r'IMG_?(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number found, place at the end


# Read the file line by line
input_file = 'data/labels_female.json'
output_file = 'data/labels_female_sorted.json'

# Load and process all entries
entries = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line.strip())
            entries.append(item)

# Update paths and sort
for item in entries:
    if 'image' in item:
        old_path = item['image']
        filename = os.path.basename(old_path)
        name_without_ext = os.path.splitext(filename)[0]
        new_filename = f"{name_without_ext}_male.jpg"
        item['image'] = f"./data/resize_dataset/{new_filename}"

# Sort entries by image number
sorted_entries = sorted(
    entries, key=lambda x: extract_image_number(x['image']))

# Save the sorted JSON file
# Option 1: Save as a JSON array
with open(output_file, 'w') as f:
    json.dump(sorted_entries, f, indent=2)

# Option 2: Save in the original NDJSON format (one JSON per line)
# with open('data/labels_female_sorted_ndjson.json', 'w') as f:
#     for item in sorted_entries:
#         json.dump(item, f)
#         f.write('\n')

# Print statistics
print(f"Processed {len(entries)} items")
print(
    f"Sorted entries from {sorted_entries[0]['image']} to {sorted_entries[-1]['image']}")

# Show first and last few entries
print("\nFirst 5 entries:")
for i, item in enumerate(sorted_entries[:5]):
    print(f"  {i+1}: {item['image']}")

print("\nLast 5 entries:")
for i, item in enumerate(sorted_entries[-5:]):
    print(f"  {len(sorted_entries)-4+i}: {item['image']}")
