import argparse
import utils


parser = argparse.ArgumentParser(description='Training and saving a neural network with userdefined hyperparameters and architecture')
parser.add_argument('data_directory', help='data directory with folders with train, valid and test(required)')
parser.add_argument('--save_dir', help='directory to save the trained neural network. [default: ""]')
parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
parser.add_argument('--learning_rate', help='learning rate for the model. [default: 0.008]')
parser.add_argument('--hidden_units', help='number of hidden units in the model. [default: 512]')
parser.add_argument('--epochs', help='number of epochs. [default: 3]')
parser.add_argument('--gpu',action='store_true', help='if gpu is available for tasks')
args = parser.parse_args()


output_path = args.save_dir if args.save_dir is not None else ''
model_type = args.arch if args.arch is not None else 'vgg19'
lr_rate = float(args.learning_rate) if args.learning_rate is not None else 0.0008
hidden_units = int(args.hidden_units) if args.hidden_units is not None else 512
num_epochs = int(args.epochs) if args.epochs is not None else 3
use_gpu = bool(args.gpu) if args.gpu is not None else False

#TRANSFORMING AND LOADING DATA
train_loader, valid_loader, test_loader = utils.process_load_data(args.data_directory)

#building the model
model = utils.build_model(model_type, hidden_units)

#TRAINING THE MODEL AND PRINTING MODEL EVALUATION WITH TRAINING LOGS
trained_model = utils.train_model(model, train_loader, valid_loader,test_loader, num_epochs, lr_rate, use_gpu)

#SAVING CHECKPOINT
utils.save_checkpoint(model, output_path, model_type, hidden_units, num_epochs, lr_rate)
