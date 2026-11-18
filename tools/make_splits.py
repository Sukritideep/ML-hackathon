
import json
from pathlib import Path

# Dataset location
DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

train_dir = DATA_ROOT / "train"
student_test_dir = DATA_ROOT / "student_test"

# Check dataset folders
if not train_dir.is_dir():
    raise FileNotFoundError(f"Train folder not found: {train_dir}")

if not student_test_dir.is_dir():
    raise FileNotFoundError(f"Student-test folder not found: {student_test_dir}")

# Collect trip IDs from existing folders
train_ids = sorted(
    folder.name for folder in train_dir.iterdir()
    if folder.is_dir()
)

student_test_ids = sorted(
    folder.name for folder in student_test_dir.iterdir()
    if folder.is_dir()
)

# Display counts
print("Training trips:", len(train_ids))
print("Student-test trips:", len(student_test_ids))

# Verify expected counts
if len(train_ids) != 80:
    raise ValueError("Expected 80 training trips.")

if len(student_test_ids) != 10:
    raise ValueError("Expected 10 student-test trips.")

# Verify no overlap
if set(train_ids) & set(student_test_ids):
    raise ValueError("Duplicate trip IDs found across partitions.")

# Create splits.json in repository root
splits = {
    "train": train_ids,
    "student_test": student_test_ids
}

with open("splits.json", "w") as file:
    json.dump(splits, file, indent=4)

print("\nsplits.json created successfully!")
