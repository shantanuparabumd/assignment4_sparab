import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir,viz_cls,rotate_point_cloud
from data_loader import get_data_loader
import os
import random
from tqdm.auto import tqdm


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/classification')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    
    parser.add_argument('--main_dir', type=str, default='./data/')

    parser.add_argument('--rotate', type=int, default=0)

    parser.add_argument('--x', type=int, default=0)

    parser.add_argument('--y', type=int, default=0)

    parser.add_argument('--z', type=int, default=0)

    parser.add_argument('--class_num', type=int, default=0)

    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')

    parser.add_argument('--batch_size', type=int, default=8, help='The number of images in a batch.')

    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')


    return parser


if __name__ == '__main__':
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<512>"

    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir+'/'+args.exp_name+'/correct')
    create_dir(args.output_dir+'/'+args.exp_name+'/incorrect')

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))

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
        batch_labels = batch_labels.to(args.device).to(torch.long)

        # Predict the labels for the batch data
        with torch.no_grad():
            batch_pred_labels = torch.argmax(model(batch_testdata), dim=-1, keepdim=False)

        # Compute the Batch accuracy
        batch_test_accuracy = batch_pred_labels.eq(batch_labels.data).cpu().sum().item()/batch_labels.size()[0]
        print(f"Batch {batch_num}: Test Accuracy {batch_test_accuracy}")
        batch_num+=1

        pred_label.append(batch_pred_labels)
        test_label.append(batch_labels)
        test_data.append(batch_testdata)

    

    pred_label = torch.cat(pred_label,dim=0)
    test_label =  torch.cat(test_label,dim=0)
    test_data =  torch.cat(test_data,dim=0)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("Test accuracy: {}".format(test_accuracy))


    num_examples = 1
    desired_class = args.class_num
    class_name = ["chair","vase","lamp"]

    file_path = 'classification_experiment_results.txt'
    # Open the file in write mode ('w')
    with open(file_path, 'a') as file:
        # Write lines to the file
        file.write("------------------------------------------\n")
        file.write(f"Experiment {args.exp_name}\n")
        file.write(f"Class {class_name[args.class_num]} Number of Point {args.num_points} Rotation X:{args.x} Y:{args.y} Z:{args.z} \n")
        file.write(f"Test Accuracy: {test_accuracy}.\n")

    

    # Visualize a few random test point clouds and failed test point clouds

    incorrect_prediction_indices = torch.nonzero((pred_label.cpu() == desired_class) & (pred_label.cpu() != test_dataloader.dataset.label)).squeeze()

    incorrect_prediction_indices = incorrect_prediction_indices.tolist()
    incorrect_prediction_indices = random.sample(incorrect_prediction_indices, num_examples)

    for ind in tqdm(incorrect_prediction_indices):
        verts = test_data[ind].detach().cpu()
        gt_cls = test_label[ind].detach().cpu().data
        pred_cls = pred_label[ind].detach().cpu().data
        path = f"output/classification/{args.exp_name}/incorrect/classification_{ind}_gt_{class_name[gt_cls]}_pred_{class_name[pred_cls]}.gif"
        viz_cls(verts, path,class_name[pred_cls] , "cuda")

    # Visualize a few random test point clouds and correct test point clouds

    correct_prediction_indices = torch.nonzero((pred_label.cpu() == desired_class) & (pred_label.cpu() == test_dataloader.dataset.label)).squeeze()
    
    correct_prediction_indices = correct_prediction_indices.tolist()
    correct_prediction_indices = random.sample(correct_prediction_indices, num_examples)

    for ind in tqdm(correct_prediction_indices):
        verts = test_data[ind].detach().cpu()
        gt_cls = test_label[ind].detach().cpu().data
        pred_cls = pred_label[ind].detach().cpu().data
        path = f"output/classification/{args.exp_name}/correct/classification_{ind}_gt_{class_name[gt_cls]}_pred_{class_name[pred_cls]}.gif"
        viz_cls(verts, path,class_name[pred_cls] ,"cuda")

