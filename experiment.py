import datetime
import argparse
from dataloader import DualLoader
import numpy as np
from encoder import *
from simsiam import *
from classifier import *
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate


# decide on set of hyperparams

# train model end to end with hyperparams

# eval model with hyper params

def load_dataloader(X_path, y_path, batch_size=64, target_size=(28,28), subset_percentage=1.0):
    # load dataset
    start_time = time.time()
    dataset = DualLoader(X_path,y_path,target_size=target_size)    
    

    # subsample dataset for data loader
    subset_percentage = args.subset_percentage
    num_samples = int(len(dataset) * subset_percentage)

    loader = DataLoader(Subset(dataset,range(num_samples)), batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"{len(loader.dataset)} records loaded in {round(time.time()-start_time,2)} seconds")
    print()
    return loader


# change to load data inside function so can cross test on 
def eval_models(encoder,classifier,device,target_shape=(28,28)):
    encoder.eval(),classifier.eval()



def main(args):
    
    # set default device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running torch on {device}"),print()

    # create dirs for logging if not present
    if 'experiments' not in os.listdir():
        os.makedirs('experiments')
    if 'models' not in os.listdir('experiments'):
        os.makedirs('experiments/model')

    
    # see if run with same id already exists
    response=None
    if str(args.run_id) in os.listdir('experiments/model/'):
        
        # if default (no manual input), just increment
        if args.run_id == 0:
            while str(args.run_id) in os.listdir('experiments/model/'):
                args.run_id += 1
        
        # if named run, prompt user for inputs
        else:
            print(f'Run ID \"{args.run_id}\" already in use')
            print('Would you like to overwrite? [[y]/n]')
            response = input()
            if response == 'n':
                print('quitting...')
                quit()
            print("overwriting...")

    if response==None: os.makedirs(f'experiments/model/{args.run_id}')

    # data paths
    data_dir = "/scratch/fc1132/capstone_data"
    X_path_train = f"{data_dir}/X_train_split.csv"
    y_path_train = f"{data_dir}/y_train_split.csv"
    X_path_val = f"{data_dir}/X_val_split.csv" 
    y_path_val = f"{data_dir}//y_val_split.csv" 
    X_path_test = f"{data_dir}//X_test_sat6.csv"
    y_path_test = f"{data_dir}//y_test_sat6.csv"

    # SELF-SUPERVISED
    # train/val data for encoder/decoder
    print(f"Creating self-supervised training DataLoader: batch size = {args.batch_size}, target size = {args.encoder_image_shape}, subset % = {round(args.subset_percentage*100,2)}")
    train_loader = load_dataloader(X_path_train,y_path_train,batch_size=args.batch_size,
                                   target_size=args.encoder_image_shape,
                                   subset_percentage=args.subset_percentage)
    print(f"Creating self-supervised validation DataLoader: batch size = {args.batch_size}, target size = {args.encoder_image_shape}, subset % = {round(args.subset_percentage*100,2)}")
    val_loader = load_dataloader(X_path_val,y_path_val,batch_size=args.batch_size,
                                   target_size=args.encoder_image_shape,
                                   subset_percentage=args.subset_percentage)

    # load and train autoencoder or simsiam
    if args.self_supervised_model=='autoencoder':
        if args.latent_shape=='conv':
            autoencoder = Autoencoder(args.latent_dim,args.encoder_image_shape)
            autoencoder.to(device)
        if args.latent_shape=='flat':
            autoencoder = FlatAutoencoder(args.encoder_image_shape[0],args.latent_dim,4)
        train_autoencoder(autoencoder,train_loader,val_loader,args.epochs,loss=args.encoder_loss, run_id=args.run_id)
        torch.save(autoencoder,f'experiments/Autoencoder/model/{args.run_id}/autoencoder_final.pth')
    if args.self_supervised_model=='simsiam':
        # TODO: incoporate conv autoencoder
        autoencoder = FlatAutoencoder(args.encoder_image_shape[0],args.latent_dim,4)
        simsiam = train_simsiam(autoencoder, train_loader, val_loader, num_epochs=args.epochs, run_id=args.run_id)
        torch.save(simsiam, f'experiments/SimSiam/model/{args.run_id}/SimSiam_final.pth')


    # CLASSIFICATION
    # train/val data for classifier
    print(f"Creating classifier training DataLoader: batch size = {args.batch_size}, target size = {args.classifier_image_shape}, subset % = {round(args.subset_percentage*100,2)}")
    train_loader = load_dataloader(X_path_train,y_path_train,batch_size=args.batch_size,
                                   target_size=args.classifier_image_shape,
                                   subset_percentage=args.subset_percentage)
    print(f"Creating classifier validation DataLoader: batch size = {args.batch_size}, target size = {args.classifier_image_shape}, subset % = {round(args.subset_percentage*100,2)}")
    val_loader = load_dataloader(X_path_val,y_path_val,batch_size=args.batch_size,
                                   target_size=args.classifier_image_shape,
                                   subset_percentage=args.subset_percentage)


    # train classifier with command line args
    if args.latent_shape=='conv':
        # determine conv input shape
        for images,y in train_loader:
            inputs,_,_=images
            inputs = inputs.to(device)
            embeddings = autoencoder.encoder(inputs)
            latent_shape = (embeddings.shape[2],embeddings.shape[3],embeddings.shape[1])
        classifier = ConvClassifier(latent_shape[0],latent_shape[1],in_channels=latent_shape[2],out_channels=latent_shape[2], dropout_prob=args.dropout)
    elif args.latent_shape=='flat':
        classifier = Classifier(input_dim=args.latent_dim,dropout_prob=args.dropout)
    
    classifier.to(device)
    if args.self_supervised_model=='autoencoder':
        train_classifier(autoencoder,classifier,train_loader,val_loader,args.epochs,run_id=args.run_id)
        torch.save(classifier,f"experiments/Autoencoder/model/{args.run_id}/classifier_final.pth")
    if args.self_supervised_model=='simsiam':
        train_classifier(simsiam,classifier,train_loader,val_loader,args.epochs,run_id=args.run_id)
        torch.save(classifier,f"experiments/SimSiam/model/{args.run_id}/classifier_final.pth")

    # eval classifier on test set
    print(f"Creating classifier test DataLoader: batch size = {args.batch_size}, target size = {args.classifier_image_shape}, subset % = {round(args.subset_percentage*100,2)}")
    test_loader = load_dataloader(X_path_test,y_path_test,batch_size=args.batch_size,
                                   target_size=args.classifier_image_shape,
                                   subset_percentage=args.subset_percentage)

    if args.self_supervised_model=='autoencoder':
        test_accuracy = evaluate_accuracy(autoencoder,classifier,test_loader)
    if args.self_supervised_model=='simsiam':
        test_accuracy = evaluate_accuracy(simsiam, classifier,test_loader)

    print(test_accuracy)

    # df to hold outputs
    df = pd.DataFrame(columns=['Timestamp','Run ID/Name','Epochs','Batch Size','Subset %','Latent Dimension Depth','Latent Dimension Output Shape','Encoder Training Dimensions','Self-Supervised Model','Autoencoder Loss Function','Classifier Training Dimensions','Classifier Test Accuracy'])
    if "run_logs.csv" not in os.listdir("experiments/{args.self_supervised_model}/"):    
        df.to_csv("experiments/{args.self_supervised_model}/run_logs.csv",index=False)
    
    # conditional for encoder loss function
    if args.self_supervised_model!='autoencoder':
        encoder_loss='N/A'
    else:
        encoder_loss = args.encoder_loss


    # add entry to file
    entry = [datetime.datetime.fromtimestamp(time.time()).strftime('%c'),args.run_id,args.epochs,args.batch_size,args.subset_percentage,args.latent_dim,args.latent_shape,args.encoder_image_shape,args.self_supervised_model,encoder_loss,args.classifier_image_shape,round(test_accuracy,2)]
    df.loc[len(df.index)] = entry
    df.to_csv("experiments/{args.self_supervised_model}/run_logs.csv",index=False,header=False,mode='a')

    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stock autoencoder on a test data set for benchmarking")

    # shared arguments
    parser.add_argument("--run_id", type=str, default=0, help="id / name for models to be stored as")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for")
    parser.add_argument("--subset_percentage", type=float, default=1.0, help="0.0-1.0 percentage of data to actually train on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader")
    parser.add_argument("--latent_dim", type=int, default=128, help="Depth of the latent dimension (encoder outputs)")
    parser.add_argument("--latent_shape", type=str, default="conv", help="Shape of the latent dimension - \"conv\" or \"flat\"")
    
    # encoder arguments
    parser.add_argument("--encoder_image_shape", type=int, nargs=2, default=(28,28), help="Shape of encoder training images")
    parser.add_argument("--self_supervised_model", type=str, default='autoencoder', help="Which self-supervised model (simsiam or autoencoder) to use in training the encoder portion")
    parser.add_argument("--encoder_loss", type=str, default="binary_crossentropy", help="Loss function to train on (binary_crossentropy or mse)")
    # NOT YET IMPLEMENTED parser.add_argument("--encoder_model_path", type=str, default=None, help="Model path to use a pretrained encoder")

    # classifier arguments
    parser.add_argument("--classifier_image_shape", type=int, nargs=2, default=(28,28), help="Shape of encoder input images during classifier training")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate for classifier")

    args = parser.parse_args()

    # run main
    main(args)
