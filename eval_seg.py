import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_point_cloud
import random
from tqdm.auto import tqdm


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/segmentation')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')

    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')

    parser.add_argument('--batch_size', type=int, default=8, help='The number of images in a batch.')

    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--rotate', type=int, default=0)

    parser.add_argument('--x', type=int, default=0)

    parser.add_argument('--y', type=int, default=0)

    parser.add_argument('--z', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir+'/'+args.exp_name+'/correct')
    create_dir(args.output_dir+'/'+args.exp_name+'/incorrect')

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    # Iterate batch in test_data
    test_dataloader = get_data_loader(args=args,train=False)

    pred_label = []
    test_label = []
    test_data = []

    batch_num = 1
    # Iterate through the batches to compute the test accuracy
    for batch in test_dataloader:
        # Get batch data and labels
        batch_testdata, batch_labels = batch
        if args.rotate:
            batch_testdata = rotate_point_cloud(batch_testdata,[args.x,args.y,args.z])
        batch_testdata = batch_testdata[:, ind].to(args.device)
        batch_labels = batch_labels[:,ind].to(args.device).to(torch.long)

        # Predict the labels for the batch data
        with torch.no_grad():
            batch_pred_labels = torch.argmax(model(batch_testdata), dim=-1, keepdim=False)

        # Compute the Batch accuracy
        batch_test_accuracy = batch_pred_labels.eq(batch_labels.data).cpu().sum().item()/batch_labels.view([-1,1]).size()[0]
        print(f"Batch {batch_num}: Test Accuracy {batch_test_accuracy}")
        batch_num+=1

        pred_label.append(batch_pred_labels)
        test_label.append(batch_labels)
        test_data.append(batch_testdata)

    

    pred_label = torch.cat(pred_label,dim=0)
    test_label =  torch.cat(test_label,dim=0)
    test_data =  torch.cat(test_data,dim=0)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.view([-1,1]).size()[0])
    print ("Test accuracy: {}".format(test_accuracy))

    file_path = 'segmentation_experiment_results.txt'
    # Open the file in write mode ('w')
    with open(file_path, 'a') as file:
        # Write lines to the file
        file.write("------------------------------------------\n")
        file.write(f"Experiment {args.exp_name}\n")
        file.write(f"Number of Point {args.num_points} Rotation X:{args.x} Y:{args.y} Z:{args.z} \n")
        file.write(f"Test Accuracy: {test_accuracy}.\n")

    num_examples = 5
    # Compute Accuracy
    # Specify the percentage of points to not match (e.g., 20%)
    non_matching_percentage = 95

    # Calculate the number of points that should not match
    non_matching_points = int(0.01 * non_matching_percentage * pred_label.size(1))

    # Find the indices where less than non_matching_points points match
    incorrect_prediction_indices = torch.nonzero((pred_label.cpu() != test_label.cpu()).sum(dim=1) > non_matching_points).squeeze()

    
    incorrect_prediction_indices = incorrect_prediction_indices.tolist()
    if len(incorrect_prediction_indices)>num_examples:
        incorrect_prediction_indices = random.sample(incorrect_prediction_indices, num_examples)
    
    sample_num = 1
    for idx in tqdm(incorrect_prediction_indices):
        verts = test_data[idx].detach().cpu()
        gt_seg = test_label[idx].detach().cpu().data
        pred_seg = pred_label[idx].detach().cpu().data
        # Visualize Segmentation Result (Pred VS Ground Truth)
        viz_seg(verts, gt_seg, "{}/{}/incorrect/gt_sample_{}.gif".format(args.output_dir, args.exp_name,sample_num), args.device,args.num_points)
        viz_seg(verts, pred_seg, "{}/{}/incorrect/pred_sample_{}.gif".format(args.output_dir, args.exp_name,sample_num), args.device,args.num_points)
        sample_num +=1

    # Visualize a few random test point clouds and correct test point clouds


    # Specify the percentage of points to match (e.g., 80%)
    matching_percentage = 50

    # Calculate the number of points that should match
    matching_points = int(0.01 * matching_percentage * pred_label.size(1))

    # Find the indices where at least matching_points points match
    correct_prediction_indices = torch.nonzero((pred_label.cpu() == test_label.cpu()).sum(dim=1) >= matching_points).squeeze()

    correct_prediction_indices = correct_prediction_indices.tolist()
    if len(correct_prediction_indices)>num_examples:
        correct_prediction_indices = random.sample(correct_prediction_indices, num_examples)

    sample_num = 1
    for idx in tqdm(correct_prediction_indices):
        verts = test_data[idx].detach().cpu()
        gt_seg = test_label[idx].detach().cpu().data
        pred_seg = pred_label[idx].detach().cpu().data
        # Visualize Segmentation Result (Pred VS Ground Truth)
        viz_seg(verts, gt_seg, "{}/{}/correct/gt_sample_{}.gif".format(args.output_dir, args.exp_name,sample_num), args.device, args.num_points)
        viz_seg(verts, pred_seg, "{}/{}/correct/pred_sample_{}.gif".format(args.output_dir, args.exp_name,sample_num), args.device, args.num_points)
        sample_num +=1

    