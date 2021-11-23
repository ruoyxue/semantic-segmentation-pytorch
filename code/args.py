import os
import logging
import torch
import argparse
from models import *


class Args:
    def __init__(self):
        self.args = self.get_args()
        self.log_arg()
        self.args = self.analyse_args()

    def get_args(self):
        """ Get args from shell """
        raise NotImplementedError

    def analyse_args(self):
        """ Analyse if args are valid, and replace str with object, return args """
        raise NotImplementedError

    def log_arg(self):
        """ Use logging.info to print args """
        output_str = ""
        tem = 0
        for key in vars(self.args).keys():
            output_str += (str(key) + " : " + str(vars(self.args)[key]) + "    ")
            tem += 1
            if tem % 5 == 0 and tem != 0:
                output_str += "\n"
        logging.info(output_str)


class TrainArgs(Args):
    def __init__(self):
        super().__init__()
    
    def get_args(self):
        parser = argparse.ArgumentParser(description="Training Process")
        parser.add_argument("--model", type=str, required=True, help="Model to be used")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        parser.add_argument("--n_class", type=int, default=2, help="Label categories")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
        parser.add_argument("--device", type=str, default="cpu", help="Device for training")
        parser.add_argument("--train_data_path", type=str, required=True,
                            help="Path for training data, with dir 'image' and 'label'")
        parser.add_argument("--save_model_path", type=str, required=True, help="Path for saving trained model")
        parser.add_argument("--check_point_path", type=str, default="../check_point", help="Path for saving checkpoint")
        parser.add_argument("--check_point_mode", type=str, default="none", help="two mode 'save', 'load'")
        return parser.parse_args()

    def analyse_args(self):
        # model
        valid_model = ["unet", "mlp", "fcn32s", "fcn16s", "fcn8s"]
        if self.args.model not in valid_model:
            raise AssertionError("Invalid args['model'] : " + "expect model in " +
                                 str(valid_model) + ", got '" + repr(self.args.model) + "'")
        else:
            if self.args.model == "unet":
                self.args.model = UNet()
            if self.args.model == "mlp":
                self.args.model = MLP()
            if self.args.model == "fcn32s":
                self.args.model = FCN32s(self.args.n_class)
            if self.args.model == "fcn16s":
                self.args.model = FCN16s(self.args.n_class)
            if self.args.model == "fcn8s":
                self.args.model = FCN8s(self.args.n_class)

        # check_point mode/path
        if self.args.check_point_mode not in ["save", "load", "none"]:
            raise AssertionError(
                "Invalid check_point_mode, expected {['save', 'load', 'none']}, got " + self.args.check_point_mode)
        if self.args.check_point_mode == "save":
            if not os.path.exists(os.path.dirname(self.args.check_point_path)):
                os.makedirs(os.path.dirname(self.args.check_point_path))
        elif self.args.check_point_mode == "load":
            if not os.path.exists(self.args.check_point_path):
                raise AssertionError("Checkpoint not exists")

        # device
        if self.args.device == "cpu":
            self.args.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.args.device = torch.device(self.args.device)
        else:
            raise RuntimeError("Invalid args['device'] : got '" + repr(self.args.device) + "'")

        # train_data_path
        if not os.path.exists(os.path.join(self.args.train_data_path, "image")) or \
            not os.path.exists(os.path.join(self.args.train_data_path, "label")):
            raise RuntimeError("Invalid train_data_path, directory '" + self.args.train_data_path +
                               "'does not includes 'image' and 'label'")
        if len(os.listdir(os.path.join(self.args.train_data_path, "image"))) == 0 or \
            len(os.listdir(os.path.join(self.args.train_data_path, "label"))) == 0:
            raise RuntimeError("Train data directory '" + self.args.train_data_path + "' is empty")

        # save_model_path
        if not os.path.exists(self.args.save_model_path):
            os.makedirs(self.args.save_model_path)
        if len(os.listdir(self.args.save_model_path)) != 0:
            raise RuntimeError("Save model directory '" + self.args.save_model_path + "' is not empty")

        return self.args


class TestArgs(Args):
    def __init__(self):
        super().__init__()
    
    def get_args(self):
        parser = argparse.ArgumentParser(description="Training Process")
        parser.add_argument("--model", type=str, required=True, help="Model to be used")
        parser.add_argument("--n_class", type=int, default=2, help="Label categories")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--chip_size", type=int, default=256, help="chip size in sliding prediction")
        parser.add_argument("--stride", type=int, default=128, help="stride in sliding prediction")

        parser.add_argument("--device", type=str, default="cpu", help="Device for training")
        parser.add_argument("--test_data_path", type=str, required=True,
                            help="Path for testing data, with dir 'image' and 'label'")
        parser.add_argument("--load_model_path", type=str, required=True, help="Path for loading trained model")
        return parser.parse_args()

    def analyse_args(self):
        # model
        valid_model = ["unet", "mlp", "fcn32s", "fcn16s", "fcn8s"]
        if self.args.model not in valid_model:
            raise AssertionError("Invalid args['model'] : " + "expect model in " +
                                 str(valid_model) + ", got '" + self.args.model + "'")
        else:
            if self.args.model == "unet":
                self.args.model = UNet()
            if self.args.model == "mlp":
                self.args.model = MLP()
            if self.args.model == "fcn32s":
                self.args.model = FCN32s(self.args.n_class)
            if self.args.model == "fcn16s":
                self.args.model = FCN16s(self.args.n_class)
            if self.args.model == "fcn8s":
                self.args.model = FCN8s(self.args.n_class)

        # device
        if torch.cuda.is_available():
            self.args.device = torch.device(self.args.device)
        else:
            raise RuntimeError("Invalid args['device'] : got '" + self.args.device + "'")

        # test_data_path
        if not os.path.exists(os.path.join(self.args.train_data_path, "image")) or \
           not os.path.exists(os.path.join(self.args.train_data_path, "label")):
            raise RuntimeError("Invalid train_data_path, directory '" + self.args.train_data_path +
                               "'does not includes 'image' and 'label'")
        if len(os.listdir(os.path.join(self.args.train_data_path, "image"))) == 0 or \
           len(os.listdir(os.path.join(self.args.train_data_path, "label"))) == 0:
            raise RuntimeError("Train data directory '" + self.args.train_data_path + "' is empty")

        # load_model_path
        if not os.path.exists(self.args.load_model_path):
            raise RuntimeError("Invalid model_path, '" + self.args.load_model_path + "' not exists")

        return self.args
