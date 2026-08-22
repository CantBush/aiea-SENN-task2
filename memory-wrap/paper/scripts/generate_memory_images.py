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

absl.flags.DEFINE_integer(
    "num_examples",
    20,
    "Number of wrong predictions to analyze"
)

absl.flags.DEFINE_integer(
    "num_random_attempts",
    1000,
    "Maximum number of random memory sets to test"
)

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
        name_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    else:
        name_classes = range(checkpoint['num_classes'])

    load_dataset = getattr(datasets, 'get_' + dataset_name)
    undo_normalization = getattr(
        datasets,
        'undo_normalization_' + dataset_name
    )

    batch_size_test = FLAGS.batch_size_test

    _, _, test_loader, mem_loader = load_dataset(
        dataset_dir,
        batch_size_train=50,
        batch_size_test=batch_size_test,
        batch_size_memory=100,
        size_train=train_examples
    )

    # ---------------------------------------------------------
    # Collect memory samples and labels
    # ---------------------------------------------------------

    memory_images = []
    memory_labels = []

    for memory_batch, memory_batch_labels in mem_loader:
        memory_images.append(memory_batch)
        memory_labels.append(memory_batch_labels)

    memory_images = torch.cat(memory_images)
    memory_labels = torch.cat(memory_labels)

    print("Memory pool size:", len(memory_images))

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------

    dir_save = (
        "../images/task7/"
        + dataset_name + "/"
        + modality + "/"
        + checkpoint['model_name'] + "/"
    )

    original_dir = os.path.join(dir_save, "original")
    correct_dir = os.path.join(dir_save, "correct_class")
    random_dir = os.path.join(dir_save, "random_success")

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)

    results = []

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image

        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))

        return transformed_im

    # ---------------------------------------------------------
    # Find and analyze wrong predictions
    # ---------------------------------------------------------

    num_wrong = 0

    for batch_idx, (images, labels) in enumerate(test_loader):

        if num_wrong >= FLAGS.num_examples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # -----------------------------------------------------
        # Original memory
        # -----------------------------------------------------

        if len(memory_images) < 100:
            print("Not enough memory samples for a memory of size 100.")
            break

        original_memory = memory_images[:100].to(device)

        outputs, rw = model(
            images,
            original_memory,
            return_weights=True
        )

        _, predictions = torch.max(outputs, 1)

        for ind in range(len(images)):

            if num_wrong >= FLAGS.num_examples:
                break

            # Only analyze WRONG predictions
            if predictions[ind] == labels[ind]:
                continue

            num_wrong += 1

            image_index = batch_idx * batch_size_test + ind

            input_selected = images[ind].unsqueeze(0)

            true_label = labels[ind].item()
            original_prediction = predictions[ind].item()

            print(
                "\nExample {}/{}".format(
                    num_wrong,
                    FLAGS.num_examples
                )
            )

            print(
                "Index: {} | Prediction: {} | True: {}".format(
                    image_index,
                    name_classes[original_prediction],
                    name_classes[true_label]
                )
            )

            # =================================================
            # 1. ORIGINAL MEMORY EXPLANATION
            # =================================================

            mem_val, memory_sorted_index = torch.sort(
                rw[ind],
                descending=True
            )

            m_ec = memory_sorted_index[mem_val > 0]

            reduced_mem = undo_normalization(
                original_memory[m_ec]
            )

            npimg = torchvision.utils.make_grid(
                reduced_mem,
                nrow=4
            ).cpu().numpy()

            fig = plt.figure(
                figsize=(batch_size_test * 2, 4),
                dpi=300
            )

            plt.subplot(1, 2, 1)

            plt.imshow(
                (get_image(input_selected) * 255)
                .astype(np.uint8),
                interpolation='nearest',
                aspect='equal'
            )

            plt.title(
                'Index: {}\nPrediction: {}\nTrue: {}'.format(
                    image_index,
                    name_classes[original_prediction],
                    name_classes[true_label]
                )
            )

            plt.axis('off')

            plt.subplot(1, 2, 2)

            plt.imshow(
                (np.transpose(npimg, (1, 2, 0)) * 255)
                .astype(np.uint8),
                interpolation='nearest',
                aspect='equal'
            )

            plt.title('Original Memory')
            plt.axis('off')

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    original_dir,
                    str(image_index) + ".png"
                )
            )

            plt.close()

            # =================================================
            # 2. CORRECT-CLASS MEMORY
            # =================================================

            correct_indices = torch.where(
                memory_labels == true_label
            )[0]

            correct_memory = None
            correct_prediction = None
            corrected_by_class = False

            if len(correct_indices) > 0:

                if len(correct_indices) >= 100:
                    selected = correct_indices[
                        torch.randperm(len(correct_indices))[:100]
                    ]
                else:
                    # Sample with replacement if necessary
                    selected = correct_indices[
                        torch.randint(
                            len(correct_indices),
                            (100,)
                        )
                    ]

                correct_memory = memory_images[selected].to(device)

                correct_outputs, correct_rw = model(
                    input_selected,
                    correct_memory,
                    return_weights=True
                )

                _, correct_pred = torch.max(
                    correct_outputs,
                    1
                )

                correct_prediction = correct_pred.item()

                corrected_by_class = (
                    correct_prediction == true_label
                )

                print(
                    "Correct-class memory prediction: {} ({})".format(
                        name_classes[correct_prediction],
                        "CORRECT" if corrected_by_class else "WRONG"
                    )
                )

                # Explanation
                mem_val, memory_sorted_index = torch.sort(
                    correct_rw[0],
                    descending=True
                )

                m_ec = memory_sorted_index[mem_val > 0]

                reduced_mem = undo_normalization(
                    correct_memory[m_ec]
                )

                npimg = torchvision.utils.make_grid(
                    reduced_mem,
                    nrow=4
                ).cpu().numpy()

                fig = plt.figure(
                    figsize=(batch_size_test * 2, 4),
                    dpi=300
                )

                plt.subplot(1, 2, 1)

                plt.imshow(
                    (get_image(input_selected) * 255)
                    .astype(np.uint8),
                    interpolation='nearest',
                    aspect='equal'
                )

                plt.title(
                    'Index: {}\nPrediction: {}\nTrue: {}'.format(
                        image_index,
                        name_classes[correct_prediction],
                        name_classes[true_label]
                    )
                )

                plt.axis('off')

                plt.subplot(1, 2, 2)

                plt.imshow(
                    (np.transpose(npimg, (1, 2, 0)) * 255)
                    .astype(np.uint8),
                    interpolation='nearest',
                    aspect='equal'
                )

                plt.title('Correct-Class Memory')
                plt.axis('off')

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        correct_dir,
                        str(image_index) + ".png"
                    )
                )

                plt.close()

            # =================================================
            # 3. RANDOM MEMORY SEARCH
            # =================================================

            random_correct = False
            random_attempt = None
            random_memory = None
            random_rw = None
            random_prediction = None

            for attempt in range(1, FLAGS.num_random_attempts + 1):

                random_indices = torch.randperm(
                    len(memory_images)
                )[:100]

                random_memory = memory_images[
                    random_indices
                ].to(device)

                random_outputs, random_rw = model(
                    input_selected,
                    random_memory,
                    return_weights=True
                )

                _, random_pred = torch.max(
                    random_outputs,
                    1
                )

                random_prediction = random_pred.item()

                if random_prediction == true_label:

                    random_correct = True
                    random_attempt = attempt

                    print(
                        "Random memory corrected prediction "
                        "on attempt {}.".format(attempt)
                    )

                    break

            if not random_correct:
                print(
                    "No successful random memory found "
                    "after {} attempts.".format(
                        FLAGS.num_random_attempts
                    )
                )

            # =================================================
            # RANDOM MEMORY EXPLANATION
            # =================================================

            if random_correct:

                mem_val, memory_sorted_index = torch.sort(
                    random_rw[0],
                    descending=True
                )

                m_ec = memory_sorted_index[mem_val > 0]

                reduced_mem = undo_normalization(
                    random_memory[m_ec]
                )

                npimg = torchvision.utils.make_grid(
                    reduced_mem,
                    nrow=4
                ).cpu().numpy()

                fig = plt.figure(
                    figsize=(batch_size_test * 2, 4),
                    dpi=300
                )

                plt.subplot(1, 2, 1)

                plt.imshow(
                    (get_image(input_selected) * 255)
                    .astype(np.uint8),
                    interpolation='nearest',
                    aspect='equal'
                )

                plt.title(
                    'Index: {}\nPrediction: {}\nTrue: {}'.format(
                        image_index,
                        name_classes[random_prediction],
                        name_classes[true_label]
                    )
                )

                plt.axis('off')

                plt.subplot(1, 2, 2)

                plt.imshow(
                    (np.transpose(npimg, (1, 2, 0)) * 255)
                    .astype(np.uint8),
                    interpolation='nearest',
                    aspect='equal'
                )

                plt.title(
                    'Random Memory\nAttempt {}'.format(
                        random_attempt
                    )
                )

                plt.axis('off')

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        random_dir,
                        str(image_index) + ".png"
                    )
                )

                plt.close()

            # =================================================
            # STORE RESULTS
            # =================================================

            results.append([
                image_index,
                true_label,
                original_prediction,
                correct_prediction,
                corrected_by_class,
                random_correct,
                random_attempt
            ])

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = os.path.join(
        dir_save,
        "task7_results.csv"
    )

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_index",
            "true_label",
            "original_prediction",
            "correct_class_prediction",
            "corrected_by_correct_class",
            "random_corrected",
            "random_attempt"
        ])

        writer.writerows(results)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    correct_class_count = sum(
        row[4] for row in results
    )

    random_count = sum(
        row[5] for row in results
    )

    print("\n========================================")
    print("TASK 7 SUMMARY")
    print("========================================")

    print(
        "Wrong predictions analyzed: {}".format(
            len(results)
        )
    )

    print(
        "Corrected with correct-class memory: {}/{}".format(
            correct_class_count,
            len(results)
        )
    )

    print(
        "Corrected with random memory: {}/{}".format(
            random_count,
            len(results)
        )
    )

    print(
        "Results saved to: {}".format(
            csv_path
        )
    )


def main(argv):
    run(FLAGS.path_model, FLAGS.dir_dataset)


if __name__ == '__main__':
    absl.app.run(main)