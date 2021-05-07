# ------------------------
#   USAGE
# ------------------------
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml
# --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml
# --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_eyes.xml

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import re

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True, help="path output data split XML file")
args = vars(ap.parse_args())

# In the iBUG 300-W dataset, each (x, y)-coordinate maps to a specific facial feature (i.e., eye, mouth, nose, etc.)
# -- in order to train a dlib shape predictor on *just* the eyes, we must first define the integer indexes
# that belong to the eyes
LANDMARKS = set(list(range(36, 48)))

# To easily parse out the eye locations from the XML file we can utilize regular expressions to determine
# if there is a 'part' element on any given line
PART = re.compile("part name='[0-9]+'")

# Load the contents of the original XML file and open the output file for writing
print("[INFO] Parsing the data split XML file...")
rows = open(args["input"]).read().strip().split("\n")
output = open(args["output"], "w")

# Loop over the rows of the data split file
for row in rows:
    # Check to see if the current line has the (x, y) coordinates for the facial landmarks we are interested in
    parts = re.findall(PART, row)
    # If there is no information related to the (x, y) coordinates of the facial landmarks we can write the current
    # line out to disk with no further modifications
    if len(parts) == 0:
        output.write("{}\n".format(row))
    # Otherwise, there is annotation information that we must process
    else:
        # Parse out the name of the attribute from the row
        attribute = "name='"
        i = row.find(attribute)
        j = row.find("'", i + len(attribute) + 1)
        name = int(row[i + len(attribute):j])
        # If the facial landmark name exists within the range of our indexes and write it to the output file
        if name in LANDMARKS:
            output.write("{}\n".format(row))
# Close the output file
output.close()
