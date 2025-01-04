import os
import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define age groups
age_groups = {
    "0-2": (0, 2),
    "3-12": (3, 12),
    "13-17": (13, 17),
    "18-59": (18, 59),
    "60+": (60, 200)
}

# Variables to keep track of male and female counts in each age group
male_counts = {age_group: 0 for age_group in age_groups.keys()}
female_counts = {age_group: 0 for age_group in age_groups.keys()}

# Load the age and gender prediction model
age_gender_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'age_net.caffemodel')

# Image directory path
image_directory = 'stats/'

# Get the list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

# Process each image in the directory
for image_file in image_files:
    # Construct the image path
    image_path = os.path.join(image_directory, image_file)

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = image[y:y + h, x:x + w].copy()

        # Preprocess the face for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(0, 0, 0), swapRB=False)

        # Print blob shape for debugging
        print("Blob Shape:", blob.shape)

        # Make predictions for age and gender
        age_gender_model.setInput(blob)
        predictions = age_gender_model.forward()

        # Get the predicted gender ('M' or 'F')
        gender_index = predictions[0].argmax()
        gender = "M" if gender_index == 0 else "F"

        # Get the predicted age group
        age_index = predictions[1].argmax()
        age_group = None
        for group, (start, end) in age_groups.items():
            if start <= age_index <= end:
                age_group = group
                break

        # Increment the count for the detected gender and age group
        if gender == 'M':
            male_counts[age_group] += 1
        else:
            female_counts[age_group] += 1

# Write the results to a file in the output directory
output_file = os.path.join(image_directory, 'gender_age_stats.txt')

with open(output_file, 'w') as f:
    f.write("Male Counts:\n")
    for age_group, count in male_counts.items():
        f.write(f"{age_group}: {count}\n")

    f.write("\nFemale Counts:\n")
    for age_group, count in female_counts.items():
        f.write(f"{age_group}: {count}\n")
