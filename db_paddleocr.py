import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import time
import json
import csv
import argparse

# Function to convert a point to a tuple
def tup(point):
    return (point[0], point[1])

# Function to check if two boxes overlap
def overlap(source, target):
    tl1, br1 = source
    tl2, br2 = target
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True

# Function to get all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps

# Step 1: Convert PDF files to images
def convert_pdfs_to_images(input_folder, output_folder):
    pdf_files = [file for file in os.listdir(input_folder) if file.lower().endswith(".pdf")]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        images = convert_from_path(pdf_path)

        # for i, image in enumerate(images):
        #     image_path = os.path.join(output_folder, f"{pdf_file}_{i + 1}.jpg")
        #     image.save(image_path, format="JPEG")

        for i, image in enumerate(images):
            # Convert PIL image to NumPy array
            image_np = np.array(image)

            # Apply histogram equalization to each channel
            eq_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            eq_image[:, :, 0] = cv2.equalizeHist(eq_image[:, :, 0])
            eq_image = cv2.cvtColor(eq_image, cv2.COLOR_LAB2RGB)

            image_path = os.path.join(output_folder, f"{pdf_file}_{i + 1}.jpg")
            cv2.imwrite(image_path, eq_image)

# Step 2: Text detection with PaddleOCR and draw bounding boxes
def text_detection_and_draw_boxes(image_path, output_folder):
    ocr = PaddleOCR(lang="en")
    result = ocr.ocr(image_path)
    input_image = cv2.imread(image_path)

    for line in result:
        box = line[0]
        top_left = tuple(map(int, box[0]))
        bottom_right = tuple(map(int, box[2]))
        cv2.rectangle(input_image, top_left, bottom_right, (0, 255, 0), 2)

    text_detection_image_path = os.path.join(output_folder, f"text_detection_{os.path.basename(image_path)}")
    cv2.imwrite(text_detection_image_path, input_image)
    print(f"Text_detection image saved at: {text_detection_image_path}")

# Step 3: Merge nearby bounding boxes and text recognition
def merge_nearby_boxes_and_text_recognition(image_path, output_folder):
    img = cv2.imread(image_path)
    orig = np.copy(img)
    blue, green, red = cv2.split(img)

    def median_canny(img, thresh1, thresh2):
        median = np.median(img)
        img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
        return img

    blue_edges = median_canny(blue, 0, 1)
    green_edges = median_canny(green, 0, 1)
    red_edges = median_canny(red, 0, 1)

    edges = blue_edges | green_edges | red_edges

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    boxes = []
    for component in zip(contours, hierarchy[0]):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0:
            boxes.append([[x, y], [x + w, y + h]])

    # Filter out excessively large boxes
    filtered = []
    max_area = 30000
    for box in boxes:
        w = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
        if w * h < max_area:
            filtered.append(box)
    boxes = filtered

    # Box merging and processing
    x_merge_margin = 15
    y_merge_margin = 30
    finished = False
    highlight = [[0, 0], [1, 1]]
    points = [[[0, 0]]]

    while not finished:
        finished = True

        index = len(boxes) - 1
        while index >= 0:
            curr = boxes[index]

            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= x_merge_margin
            tl[1] -= y_merge_margin
            br[0] += x_merge_margin
            br[1] += y_merge_margin

            overlaps = getAllOverlaps(boxes, [tl, br], index)

            if len(overlaps) > 0:
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br = boxes[ind]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)

                x, y, w, h = cv2.boundingRect(con)
                w -= 1
                h -= 1
                merged = [[x, y], [x + w, y + h]]

                highlight = merged[:]
                points = con

                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]
                boxes.append(merged)

                finished = False
                break

            index -= 1

    final_copy = np.copy(orig)
    for box in boxes:
        cv2.rectangle(final_copy, tup(box[0]), tup(box[1]), (0, 200, 0), 1)

    merge_bbox_path = os.path.join(output_folder, f"merge_bbox_{os.path.basename(image_path)}")
    cv2.imwrite(merge_bbox_path, final_copy)
    print(f"Merge bounding box saved at: {merge_bbox_path}")

    left_blocks = []
    right_blocks = []

    for idx, box in enumerate(boxes):
        if box[0][0] < w / 2:
            left_blocks.append((box, idx))
        else:
            right_blocks.append((box, idx))

    left_blocks.sort(key=lambda x: x[0][0][1])
    right_blocks.sort(key=lambda x: x[0][0][1])

    merged_block_ids = [idx for _, idx in left_blocks] + [idx for _, idx in right_blocks]

    # PaddleOCR
    ocr = PaddleOCR(lang="en")

    # #json file
    # bounding_box_texts = []

    # for block_id in merged_block_ids:
    #     box = boxes[block_id]
    #     x1, y1 = box[0]
    #     x2, y2 = box[1]

    #     segment_image = img[y1:y2, x1:x2]

    #     result = ocr.ocr(segment_image)

    #     text_list = []
    #     for line in result:
    #         text = line[1][0]
    #         text_list.append(text)
        
    #     # Split the text_list into smaller lists of a specified length (e.g., 5)
    #     split_text_lists = [text_list[i:i + 5] for i in range(0, len(text_list), 5)]
        
    #     bounding_box_texts.append({
    #         "box": box,
    #         "texts": text_list
    #     })

    # # json_output_path = os.path.join(output_folder, f'bounding_boxes_{os.path.basename(image_path)}.json')

    # # with open(json_output_path, 'w') as json_file:
    # #     json.dump(bounding_box_texts, json_file, indent=4)
    
    # txt file
    output_txt_path = os.path.join(output_folder,
                                   f'recognized_text_{os.path.basename(image_path)}.txt')

    with open(output_txt_path, 'w') as txt_file:
        for block_id in merged_block_ids:
            box = boxes[block_id]
            x1, y1 = box[0]
            x2, y2 = box[1]

            segment_image = img[y1:y2, x1:x2]

            result = ocr.ocr(segment_image)

            for line in result:
                text = line[1][0]
                txt_file.write(text + '\n')

# Step 4: Process PDF files
def process_pdf_files(pdf_folder, output_folder):
    convert_pdfs_to_images(pdf_folder, output_folder)

    image_files = [file for file in os.listdir(output_folder) if file.lower().endswith(".jpg")]
    for image_file in image_files:
        image_path = os.path.join(output_folder, image_file)

        text_detection_and_draw_boxes(image_path, output_folder)
        merge_nearby_boxes_and_text_recognition(image_path, output_folder)

def save_execution_time_to_csv(start_time, end_time, csv_path):
    header = ["Date", "Total Execution Time (seconds)"]
    total_runtime = end_time - start_time
    data = [time.strftime("%Y-%m-%d %H:%M:%S"), total_runtime]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

def cli_with_argparse(args):    
    pdf_folder = args.i
    output_folder = args.o

    start_time = time.time()

    process_pdf_files(pdf_folder, output_folder)

    end_time = time.time()

    print(f"Total script execution time: {end_time - start_time:.2f} seconds")

    csv_path = os.path.join(output_folder, "execution_times.csv")
    save_execution_time_to_csv(start_time, end_time, csv_path)

parser = argparse.ArgumentParser(description="PDF Processing Script")
parser.add_argument("--i", required=True, help="Path to the input folder")
parser.add_argument("--o", required=True, help="Path to the output folder")
args = parser.parse_args()

cli_with_argparse(args)
