import sys
sys.path.append('..')
import os
import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS



def run(path:str,dataset_dir:str):
    """ Function to generate memory images for testing images using a given
    model. Memory images show the samples in the memory set that have an
    impact on the current prediction.

    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device:{}".format(device))    
    # load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)
    
    #saving stuff
    dir_save = "../images/mem_images/"+dataset_name+"/"+modality+"/" + checkpoint['model_name'] + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im


    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= 50:
            break

        print("Batch:{}/{}".format(batch_idx, len(test_loader)), end='\r')

        try:
            memory, _ = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            memory, _ = next(memory_iter)

        images = images.to(device)
        memory = memory.to(device)
        labels = labels.to(device)

        # compute output
        outputs, rw = model(images, memory, return_weights=True)
        _, predictions = torch.max(outputs, 1)

        # compute memory outputs
        mem_val, memory_sorted_index = torch.sort(rw, descending=True)

        for ind in range(len(images)):

            # Only generate explanations for WRONG predictions
            if predictions[ind] == labels[ind]:
                continue

            # Absolute index of this image in the test dataset
            image_index = batch_idx * batch_size_test + ind

            input_selected = images[ind].unsqueeze(0)

            # M_c u M_e : samples with a positive impact on prediction
            m_ec = memory_sorted_index[ind][mem_val[ind] > 0]

            # get reduced memory
            reduced_mem = undo_normalization(memory[m_ec])
            npimg = torchvision.utils.make_grid(
                reduced_mem,
                nrow=4
            ).cpu().numpy()

            # Create figure
            fig = plt.figure(
                figsize=(batch_size_test * 2, 4),
                dpi=300
            )

            # Input image
            plt.subplot(1, 2, 1)

            plt.imshow(
                (get_image(input_selected) * 255).astype(np.uint8),
                interpolation='nearest',
                aspect='equal'
            )

            plt.title(
                'Index: {}\nPrediction: {}\nTrue: {}'.format(
                    image_index,
                    name_classes[predictions[ind]],
                    name_classes[labels[ind]]
                )
            )

            plt.axis('off')

            # Memory samples
            plt.subplot(1, 2, 2)

            plt.imshow(
                (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8),
                interpolation='nearest',
                aspect='equal'
            )

            plt.title('Used Samples')
            plt.axis('off')

            fig.tight_layout()

            # Save using the absolute test-set index
            fig.savefig(
                dir_save + str(image_index) + ".png"
            )

            plt.close()

        print(
            'Processed {}/{} batches'.format(
                batch_idx,
                len(test_loader)
            ),
            end='\r'
        )


def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)