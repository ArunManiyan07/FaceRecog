import cv2
import os
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Path to your images folder
image_folder = "images"

# Create output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Loop through images
for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, file_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ Could not load {file_name}")
            continue

        # Detect faces
        results = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print(f"🔍 Found {len(results)} face(s) in {file_name}")

        # Draw bounding boxes
        for face in results:
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Save result
        save_path = os.path.join(output_folder, f"detected_{file_name}")
        cv2.imwrite(save_path, image)
        print(f"✅ Saved: {save_path}")

        # Show in original size
        cv2.imshow("Face Detection (MTCNN)", image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
print("🎉 Detection finished! Check 'output/' folder for results.")
