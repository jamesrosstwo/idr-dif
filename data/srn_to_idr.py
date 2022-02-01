import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


# From the SRN Repository: https://github.com/vsitzmann/scene-representation-networks/blob/8165b500816bb1699f5a34782455f2c4b6d4f35a/util.py#L44
def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False, img_width=None, img_height=None):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())
        if img_width:
            width = img_width
        if img_height:
            height = img_height

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


def convert_car_by_id(id, base_path="cars_train", n_views=250, delete_existing=False, desired_img_shape=None):
    in_path = Path(base_path) / id

    # --- Create output paths ---
    base_out_path = Path('idr') / id

    if delete_existing and base_out_path.exists():
        shutil.rmtree(str(base_out_path))

    images_path = base_out_path / "image"
    mask_path = base_out_path / "mask"
    base_out_path.mkdir(parents=True)
    images_path.mkdir()
    mask_path.mkdir()

    # --- Camera Information ---
    full_intrinsic, grid_barycenter, scale, world2cam_poses = parse_intrinsics(in_path / "intrinsics.txt",
                                                                               img_width=desired_img_shape[0],
                                                                               img_height=desired_img_shape[1])
    cam_npz = dict()

    for i in range(n_views):
        filled_id = str(i).zfill(6)
        img_filename = filled_id + ".png"
        rgb_path = in_path / "rgb" / img_filename
        img = cv2.imread(str(rgb_path))

        # Resize to correct shape
        if desired_img_shape:
            min_resize_dims = (min(desired_img_shape), min(desired_img_shape))
            img = cv2.resize(img, min_resize_dims)
            pad_amt = (max(desired_img_shape) - min(desired_img_shape)) / 2.0
            pad_amts = int(math.floor(pad_amt)), int(math.ceil(pad_amt))
            pad_axis = np.argmax(desired_img_shape)
            if pad_axis == 0:
                img = cv2.copyMakeBorder(img, 0, 0, *pad_amts, cv2.BORDER_REPLICATE)

        cv2.imwrite(str(images_path / img_filename), img)

        # Flood fill to create mask
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = im_gray.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        flooded = cv2.floodFill(im_gray, mask, (2, 2), 255)[2]
        _, mask = cv2.threshold(flooded, 0, 1, cv2.THRESH_BINARY_INV)
        mask = mask[1:-1, 1:-1] * 255

        cv2.imwrite(str(mask_path / img_filename), mask)

        pose = load_pose(in_path / "pose" / (filled_id + ".txt"))
        cam_mx = full_intrinsic @ pose
        cam_npz["world_mat_" + str(i)] = cam_mx
        # cam_npz["scale_mat_" + str(i)] = np.identity(4)

    np.savez(str(base_out_path / "cameras.npz"), **cam_npz)


if __name__ == "__main__":
    convert_car_by_id("1a1dcd236a1e6133860800e6696b8284", delete_existing=True, desired_img_shape=(1600, 1200))
