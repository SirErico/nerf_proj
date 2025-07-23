import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from glob import glob
import os
import argparse

def load_and_fix_json(input_file):
    """Naprawia niepoprawny JSON (usuwa brakujące przecinki) i zwraca dane jako listę."""
    with open(input_file, 'r') as f:
        raw = f.read()

    # Dodaj przecinki między kolejnymi obiektami
    raw = raw.replace('}\n{', '},\n{')

    # Opakuj w listę
    fixed = f'[{raw}]'

    return json.loads(fixed)


def convert_to_nerf_frames(data):
    """Konwertuje dane z listy (translation + quaternion) na format NeRF frames."""
    frames = []

    for idx, entry in enumerate(data):
        t = entry['translation']
        q = entry['rotation']

        # Zamiana kwaternionu (x, y, z, w) na macierz rotacji 3x3
        quat = [q['x'], q['y'], q['z'], q['w']]
        rot_matrix = R.from_quat(quat).as_matrix()

        # Macierz 4x4
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = [t['x'], t['y'], t['z']]

        frames.append({
            "file_path": f"./train/r_{idx}.png",
            "transform_matrix": transform_matrix.tolist()
        })

    return frames


def normalize_matrix_opencv_to_nerf(mat):
    """
    Converts a 4x4 camera pose from OpenCV (COLMAP) to NeRF/Blender coordinate system.
    - OpenCV: +X right, +Y down, +Z forward.
    - NeRF:   +X right, +Y up, +Z backward (camera looks along -Z).
    Also orthonormalizes the rotation to remove numerical errors.
    """
    mat = np.array(mat)
    R = mat[:3, :3]
    t = mat[:3, 3]

    # Orthonormalizacja R
    u, _, vh = np.linalg.svd(R)
    R = np.dot(u, vh)

    # Złóż macierz na nowo
    fixed_mat = np.eye(4)
    fixed_mat[:3, :3] = R
    fixed_mat[:3, 3] = t

    # Przejście z układu OpenCV do NeRF (flipy Y i Z)
    opencv_to_nerf = np.diag([-1, 1, -1, 1])
    fixed_mat = fixed_mat @ opencv_to_nerf

    return fixed_mat.tolist()


def process_json_to_nerf(input_json, train_dir, output_json):
    """
    Łączy wszystkie kroki w jeden:
    - Naprawia JSON
    - Konwertuje na format NeRF (frames)
    - Poprawia macierze transformacji (układ współrzędnych NeRF)
    - Dodaje parametry kamery
    - Zapisuje finalny transforms.json
    """
    # Wczytaj i napraw JSON
    raw_data = load_and_fix_json(input_json)

    # Konwertuj do frames
    frames = convert_to_nerf_frames(raw_data)

    # Znajdź pliki w folderze train (sortowane)
    files = sorted(glob(os.path.join(train_dir, '*')))
    if not files:
        raise ValueError(f"Brak plików w folderze {train_dir}")

    # Przypisz pliki i popraw macierze
    for i, frame in enumerate(frames):
        if i < len(files):
            frame["file_path"] = f"./{os.path.relpath(files[i], os.path.dirname(output_json))}"
        frame["transform_matrix"] = normalize_matrix_opencv_to_nerf(frame["transform_matrix"])

    # Parametry kamery
    camera_params = {
        "camera_model": "OPENCV",
        "fl_x": 610.35,
        "fl_y": 609.99,
        "cx": 631.7,
        "cy": 353.59,
        "w": 1280,
        "h": 720,
        "frames": frames
    }

    # Zapisz finalny transforms.json
    with open(output_json, 'w') as f:
        json.dump(camera_params, f, indent=4)

    print(f"Finalny plik zapisano: {output_json}")


if __name__ == "__main__":
    # PRZYKŁAD: bez plików pośrednich
    argparser = argparse.ArgumentParser(description="Konwersja JSON do formatu NeRF.")
    argparser.add_argument("--input-json", type=str, required=True, help="Ścieżka do wejściowego pliku JSON.")
    argparser.add_argument("--train-dir", type=str, default="train", help="Folder z obrazami treningowymi.")
    argparser.add_argument("--output-json", type=str, default="transforms.json", help="Ścieżka do wyjściowego pliku transforms.json.")
    args = argparser.parse_args()
    process_json_to_nerf(args.input_json, args.train_dir , args.output_json)
