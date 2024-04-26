import cv2
import numpy as np

def convert_to_grayscale(input_image_path, output_image_path_prefix):
    # Load the image
    image = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(grayscale_image, 50, 150)

    # Invert the edges to get the lines
    lines = cv2.bitwise_not(edges)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(lines, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the tuple to a list and sort it by area
    contours_list = list(contours)
    contours_list.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # Keep only the top 3 contours
    top_contours = contours_list[:3]

    # Extract the coordinates of the bounding boxes with an additional 7 mm margin
    boxes = []
    for contour in top_contours:
        # Get the bounding box of the current contour
        x, y, w, h = cv2.boundingRect(contour)

        # Expand the bounding box by 7 mm in each direction
        x -= 10
        y -= 10
        w += 20
        h += 20

        # Add the adjusted bounding box coordinates to the list
        boxes.append((x, y, w, h))

    image_to_draw = image.copy()

    # Draw the adjusted bounding boxes on the image
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Construct the output image path
    output_image_path = f"{output_image_path_prefix}_with_boxes.png"

    # Save the main output image with adjusted bounding boxes
    cv2.imwrite(output_image_path, image_to_draw)

    # Calculate the centers of the bounding boxes
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in boxes]

    # Calculate distances from corners to centers
    distances = []
    for center in centers:
        center_x, center_y = center
        distances_from_corners = [
            np.sqrt((0 - center_x)**2 + (0 - center_y)**2), # Distance from left top corner
            np.sqrt((0 - center_x)**2 + (image.shape[0] - center_y)**2), # Distance from left bottom corner
            np.sqrt((image.shape[1] - center_x)**2 + (0 - center_y)**2) # Distance from right top corner
        ]
        distances.append(distances_from_corners)

    # Sort the distances and boxes together
    sorted_distances_boxes = sorted(list(zip(distances, boxes)), key=lambda x: (x[0][0], x[0][1], x[0][2]))

    # Extract the sorted boxes
    sorted_boxes = [b for d, b in sorted_distances_boxes]

    # Save each bounding box as a separate image
    for i, box in enumerate(sorted_boxes):
        x, y, w, h = box
        cropped_image = image[y:y+h, x:x+w]
        output_image_path = f"{output_image_path_prefix}_{i+1}.png"
        cv2.imwrite(output_image_path, cropped_image)

    # Removed cv2.waitKey(0) and cv2.destroyAllWindows() as they are not needed for saving images

input_image_path = input ("Введите путь входного изображения: ") #эта функция введена для удобства простых пользователей
#input_image_path = 'C:\\Diplom_po_ii\\t1.png' пример того что надо вводить
output_image_path_prefix = input ("Введите путь выходного массива изображений, без рассширения: ") #эта функция введена для удобства простых пользователей
#output_image_path_prefix = 'C:\\Diplom_po_ii\\t1' пример того что надо вводить

convert_to_grayscale(input_image_path, output_image_path_prefix)

#*имя файла*_1 - всегда вид спереди
#*имя файла*_2 - всегда вид сверху
#*имя файла*_3 - всегда вид справа

# написанное сверху работает правильно если учитывать что вид спериди изображается в левом верхнем углу, вид сверху в левом нижнем углу, вид справа в верхнем правом углу 