import os
import logging
import warnings

import torch
import argparse
from models import *
import copy


def set_logger(save_path):
    """ set logger which output in both terminal and save in file """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(logging.Formatter("%(message)s"))
    terminal_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(filename=save_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)


class Args:
    def __init__(self):
        self.args = self.get_args()
        self.origin = copy.deepcopy(vars(self.args))  # save original config
        # self.log_arg()
        self.args = self.analyse_args()

    def __getattr__(self, item):
        if item not in self.args.__dict__.keys():
            raise KeyError(f"invalid key for args, expects {list(self.args.__dict__.keys())}, got {item}")
        return self.args.__dict__[item]

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
        parser.add_argument("--random_seed", type=int, help="random seed")
        parser.add_argument("--n_class", type=int, default=2, help="Label categories")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
        parser.add_argument("--device", type=str, default="cpu", help="Device for training")
        parser.add_argument("--chip_size", type=int, help="chip size in sliding validation")
        parser.add_argument("--stride", type=int, help="stride in sliding validation")
        parser.add_argument("--exp_path", type=str, required=True,
                            help="Path for saving training experiment info")
        parser.add_argument("--train_data_path", type=str, required=True,
                            help="Path for training data, with dir 'image' and 'gt'")
        parser.add_argument("--valid_data_path", type=str, required=True,
                            help="Path for valid data, with dir 'image' and 'gt'")
        parser.add_argument("--check_point_mode", type=str, default="none",
                            help="two mode 'save', 'load'")
        return parser.parse_args()

    def analyse_args(self):
        # model
        valid_model = ["unet", "mlp", "fcn32s", "fcn16s", "fcn8s", "dlinknet34", "dlinknet50", "dlinknet101"]
        if self.args.model not in valid_model:
            raise AssertionError(f"Invalid args['model']: expect model in {str(valid_model)}, "
                                 f"got {repr(self.args.model)}")
        else:
            if self.args.model == "unet":
                self.args.model = UNet(n_class=self.args.n_class, n_channel=3)
            if self.args.model == "mlp":
                self.args.model = MLP()
            if self.args.model == "fcn32s":
                self.args.model = FCN32s(self.args.n_class)
            if self.args.model == "fcn16s":
                self.args.model = FCN16s(self.args.n_class)
            if self.args.model == "fcn8s":
                self.args.model = FCN8s(self.args.n_class)
            if self.args.model == "dlinknet34":
                self.args.model = DLinkNet34(self.args.n_class, pretrained=True)
            if self.args.model == "dlinknet50":
                self.args.model = DLinkNet50(self.args.n_class, pretrained=True)
            if self.args.model == "dlinknet101":
                self.args.model = DLinkNet101(self.args.n_class, pretrained=True)


        # device
        if self.args.device == "cpu":
            self.args.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.args.device = torch.device(self.args.device)
        else:
            raise RuntimeError(f"Invalid args['device'] : got {repr(self.args.device)}")

        # train_data_path
        if not os.path.exists(os.path.join(self.args.train_data_path, "image")) or \
           not os.path.exists(os.path.join(self.args.train_data_path, "gt")):
            raise FileNotFoundError("Invalid train_data_path, directory '" + self.args.train_data_path +
                                    "'does not includes 'image' and 'gt'")
        if len(os.listdir(os.path.join(self.args.train_data_path, "image"))) == 0 or \
           len(os.listdir(os.path.join(self.args.train_data_path, "gt"))) == 0:
            raise FileNotFoundError("Train data directory '" + self.args.train_data_path + "' is empty")

        # valid_data_path
        if not os.path.exists(os.path.join(self.args.valid_data_path, "image")) or \
           not os.path.exists(os.path.join(self.args.valid_data_path, "gt")):
            raise FileNotFoundError("Invalid valid_data_path, directory '" + self.args.valid_data_path +
                                    "'does not includes 'image' and 'gt'")
        if len(os.listdir(os.path.join(self.args.valid_data_path, "image"))) == 0 or \
           len(os.listdir(os.path.join(self.args.valid_data_path, "gt"))) == 0:
            raise FileNotFoundError("Valid data directory '" + self.args.valid_data_path + "' is empty")

        # exp_path
        if self.origin["model"].lower() not in os.path.basename(self.origin["exp_path"]).lower():
            raise RuntimeError("selected model({}) is not compatible with exp_path({})".
                               format(self.origin["model"], self.origin["exp_path"]))
        if self.args.check_point_mode != "load":
            if not os.path.exists(self.args.exp_path):
                os.makedirs(self.args.exp_path)
            for root, dirs, files in os.walk(os.path.join(self.args.exp_path)):
                if files:
                    raise RuntimeError("exp train path {} has files".format(self.args.exp_path))

        # save model path
        save_model_path = os.path.join(self.args.exp_path, "model_saved")
        if self.args.check_point_mode != "load":
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            if len(os.listdir(save_model_path)) != 0:
                raise RuntimeError(f"save model directory ({save_model_path}) is not empty")

        # checkpoint mode/path
        check_point_path = os.path.join(self.args.exp_path, "checkpoint_saved", "checkpoint.pt")
        if self.args.check_point_mode not in ["save", "load", "none"]:
            raise AssertionError(
                "Invalid check_point_mode, expected {['save', 'load', 'none']}, got " + self.args.check_point_mode)
        if self.args.check_point_mode == "save":
            if not os.path.exists(os.path.dirname(check_point_path)):
                os.makedirs(os.path.dirname(check_point_path))
        elif self.args.check_point_mode == "load":
            if not os.path.exists(check_point_path):
                raise AssertionError("checkpoint ({}) not exists".format(check_point_path))

        # tensorboard save path
        tensorboard_save_path = os.path.join(self.args.exp_path, "tensorboard_saved")
        if not os.path.exists(tensorboard_save_path):
            os.makedirs(tensorboard_save_path)
        if len(os.listdir(tensorboard_save_path)) != 0 and self.args.check_point_mode == "save":
            raise RuntimeError(f"save tensorboard directory ({tensorboard_save_path}) is not empty")

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
        parser.add_argument("--exp_path", type=str, required=True,
                            help="Path for saving test experiment info")
        return parser.parse_args()

    def analyse_args(self):
        # model
        valid_model = ["unet", "mlp", "fcn32s", "fcn16s", "fcn8s", "dlinknet34", "dlinknet50", "dlinknet101"]
        if self.args.model not in valid_model:
            raise AssertionError(f"Invalid args['model']: expect model in {str(valid_model)}, "
                                 f"got {repr(self.args.model)}")
        else:
            if self.args.model == "unet":
                self.args.model = UNet(n_class=self.args.n_class, n_channel=3)
            if self.args.model == "mlp":
                self.args.model = MLP()
            if self.args.model == "fcn32s":
                self.args.model = FCN32s(self.args.n_class)
            if self.args.model == "fcn16s":
                self.args.model = FCN16s(self.args.n_class)
            if self.args.model == "fcn8s":
                self.args.model = FCN8s(self.args.n_class)
            if self.args.model == "dlinknet34":
                self.args.model = DLinkNet34(self.args.n_class, pretrained=True)
            if self.args.model == "dlinknet50":
                self.args.model = DLinkNet50(self.args.n_class, pretrained=True)
            if self.args.model == "dlinknet101":
                self.args.model = DLinkNet101(self.args.n_class, pretrained=True)
        # device
        if torch.cuda.is_available():
            self.args.device = torch.device(self.args.device)
        else:
            raise AssertionError("Invalid args['device'] : got '" + repr(self.args.device) + "'")

        # test_data_path
        if not os.path.exists(os.path.join(self.args.test_data_path, "image")) or \
           not os.path.exists(os.path.join(self.args.test_data_path, "gt")):
            raise FileNotFoundError("Invalid train_data_path, directory '" + self.args.test_data_path +
                                    "'does not includes 'image' and 'gt'")
        if len(os.listdir(os.path.join(self.args.test_data_path, "image"))) == 0 or \
           len(os.listdir(os.path.join(self.args.test_data_path, "gt"))) == 0:
            raise FileNotFoundError("Test data directory '" + self.args.test_data_path + "' is empty")

        # exp_path
        if self.origin["model"].lower() not in os.path.basename(self.origin["exp_path"]).lower():
            raise RuntimeError("selected model({}) is not compatible with exp_path({})".
                               format(self.origin["model"], self.origin["exp_path"]))
        if not os.path.exists(self.args.exp_path):
            raise FileNotFoundError(f"exp test path ({self.args.exp_path}) not exist")
        if os.path.exists(os.path.join(self.args.exp_path, "log_test.txt")):
            raise RuntimeError(f"exp test path ({self.args.exp_path}) already has file log_test.txt")

        # load_model_path
        load_model_path = os.path.join(self.args.exp_path, "model_saved", "model.pth")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Invalid model_path ({load_model_path}) not exists")

        # save_output_path
        save_output_path = os.path.join(self.args.exp_path, "prediction_saved")
        if not os.path.exists(save_output_path):
            os.makedirs(save_output_path)
        if len(os.listdir(save_output_path)) != 0:
            raise RuntimeError(f"Save predictions directory ({save_output_path}) is not empty")

        return self.args
