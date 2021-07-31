import os
import shutil


# classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
#            'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


def main():
    source_root_path = 'F:/Research/PHD_AIZU/tiny_ai/ear/data/c100' # path to source data
    list_of_classes = os.listdir(os.path.join(source_root_path, 'train')) # list the avaiable classes
    destination_root_path = 'F:/Research/PHD_AIZU/tiny_ai/ear/data/c100_combined/train/' # destination 
    counter = 0 # count the number of images

    classes = {} 
    class_counter = 0
    for cls in list_of_classes:
        classes[cls] = class_counter
        class_counter = class_counter + 1
    print (classes)


    for cls in list_of_classes:
        source_root_path_for_class = os.path.join(source_root_path, 'train', cls)
        files_per_class = os.listdir(source_root_path_for_class)
        files_per_class.sort()
        for fl in files_per_class:
            id_ = classes[cls]
            new_file_name = str(id_) + '_' + str(counter) + '.png'
            destination = destination_root_path + new_file_name
            source = os.path.join(source_root_path, 'train', cls, fl)
            shutil.copy(source, destination)
            counter = counter + 1

if __name__ == '__main__':
    main()