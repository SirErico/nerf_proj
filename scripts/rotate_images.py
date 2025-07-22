import cv2
import os
from glob import glob
import argparse

def rotate_images_180(input_folder, output_folder=None):
    # Jeśli nie podano folderu wyjściowego, zapisujemy w tym samym miejscu
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)

    # Wczytaj wszystkie obrazy (png, jpg)
    image_files = glob(os.path.join(input_folder, '*.*'))

    for img_path in image_files:
        # Wczytaj obraz
        img = cv2.imread(img_path)

        if img is None:
            print(f"Nie udało się wczytać obrazu: {img_path}")
            continue

        # Obrót o 180 stopni (flip w pionie i poziomie)
        rotated = cv2.rotate(img, cv2.ROTATE_180)

        # Zapisz wynik
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, rotated)
        print(f"Zapisano obrócony obraz: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotatinf images by 180 degrees in a folder.")
    parser.add_argument("--input-path", type=str,default="train/", help="Path to the input folder containing images.")
    args = parser.parse_args()
    rotate_images_180(args.input_path)
