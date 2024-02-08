import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
from PIL import Image, ImageDraw,ImageFont
from tqdm.auto import tqdm
import numpy as np

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device,points):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,points,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

    my_images = []
    for r in rend:
        image = Image.fromarray((r * 255).astype(np.uint8))

        my_images.append(np.array(image))

    imageio.mimsave(path, my_images, loop=0)

def viz_cls (verts, path,class_name, device):
    """
    visualize classification result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    my_images = []
    
    class_color = {"chair": [1.0, 0.62, 0.027], "vase": [1.0, 0.027, 0.471], "lamp": [0.102, 0.616, 10.678]}
    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.repeat(30,1,1).to(torch.float)
    sample_colors = torch.tensor(class_color[class_name]).repeat(1,sample_verts.shape[1],1).repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

    for r in rend:
        image = Image.fromarray((r * 255).astype(np.uint8))

        #     # Create a drawing object
        # draw = ImageDraw.Draw(image)

        # # Add text to the image
        # text = class_name
        # font_size = 20
        # font_color = (0, 0, 0)  # RGB color tuple

        # font = ImageFont.truetype('font.ttf', font_size)  
        
        # text_position = (10, 10)

        # draw.text(text_position, text, fill=font_color, font=font)

        my_images.append(np.array(image))

    imageio.mimsave(path, my_images, loop=0)

    

def rotate_point_cloud(data, rotation_angles):
    """
    Rotate a 3D point cloud.

    Parameters:
    - point_cloud: np.ndarray, shape (N, 3), the 3D point cloud
    - rotation_angles: tuple of floats (angle_x, angle_y, angle_z), rotation angles in degrees

    Returns:
    - rotated_point_cloud: np.ndarray, shape (N, 3), the rotated 3D point cloud
    """

    # Convert angles from degrees to radians
    angles = np.radians(rotation_angles)

    # Define rotation matrices for x, y, z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])

    # Combine rotation matrices
    rotation_matrix = torch.tensor(Rz @ Ry @ Rx).float()

    # Apply rotation to the point cloud
    rotated_data = (rotation_matrix @ data.transpose(1, 2)).transpose(1, 2)

    return rotated_data