import os
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import torch


# From the SRN Repository: https://github.com/vsitzmann/scene-representation-networks/blob/8165b500816bb1699f5a34782455f2c4b6d4f35a/util.py#L44
def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def converted_srn(in_path: Path, out_path: Path, n_views=250, overwrite=False, img_sz=None):
    print(out_path)
    # --- Create output paths ---
    if overwrite and out_path.exists():
        shutil.rmtree(str(out_path))

    images_path = out_path / "image"
    mask_path = out_path / "mask"
    out_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    mask_path.mkdir(exist_ok=True)

    # --- Camera Information ---
    full_intrinsic, grid_barycenter, scale, world2cam_poses = parse_intrinsics(in_path / "intrinsics.txt",
                                                                               trgt_sidelength=img_sz)
    cam_npz = dict()

    for i in range(n_views):
        base_filled_id = str(i).zfill(6)
        filled_id = str(i).zfill(6)
        img_filename = "{0}.png".format(filled_id)
        rgb_path = in_path / "rgb" / (base_filled_id + ".png")
        img = cv2.imread(str(rgb_path))

        cv2.imwrite(str(images_path / img_filename), img)

        # Flood fill to create mask
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = im_gray.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        flooded = cv2.floodFill(im_gray, mask, (2, 2), 255)[2]
        _, mask = cv2.threshold(flooded, 0, 1, cv2.THRESH_BINARY_INV)
        mask = mask[1:-1, 1:-1] * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(mask_path / img_filename), mask)

        pose = load_pose(in_path / "pose" / ("{0}.txt".format(base_filled_id)))
        extrinsic = np.linalg.inv(pose)
        cam_mx = full_intrinsic @ extrinsic
        cam_npz["world_mat_{0}".format(i)] = cam_mx
        cam_npz["scale_mat_{0}".format(i)] = np.identity(4)
        np.savez(str(out_path / "cameras.npz"), **cam_npz)
    return cam_npz


def converted_srn_collection(in_path: Path, out_path: Path, n_views=250, n_objs=None, overwrite=False, img_sz=None):
    if overwrite and out_path.exists():
        shutil.rmtree(str(out_path))


    specs = dict()

    i = 0
    in_paths = list(in_path.glob("*"))
    if isinstance(n_objs, int):
        in_paths = in_paths[:n_objs]

    for sub_path in in_paths:
        if os.path.isdir(sub_path):
            print("Converting", sub_path)
            converted_srn(sub_path, out_path / "scan{0}".format(i), n_views=n_views, overwrite=False, img_sz=img_sz)
            i += 1
    specs["n_objs"] = i
    with open(str(out_path / "specs.json"), "w") as f:
        json.dump(specs, f)



if __name__ == "__main__":
    in_path = Path("cars_train")
    out_path = Path("idr/collection2")
    converted_srn_collection(in_path, out_path, overwrite=True, img_sz=128, n_objs=2)
