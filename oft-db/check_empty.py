import os
import fnmatch


subject_names = [
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can", 
    "candle", "cat", "cat2", "clock", "colorful_sneaker",    
    "dog", "dog2", "dog3", "dog5", "dog6",    
    "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",    
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",    
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie"
]



def find_subdirectory_names_without_png(directory):
    subdirectory_names_without_png = []
    
    for root, dirs, files in os.walk(directory):
        png_files = [f for f in files if fnmatch.fnmatch(f, '*.png')]
        
        if not png_files:
            subdirectory_name = os.path.basename(root)
            subdirectory_names_without_png.append(subdirectory_name)
    
    all_contents = os.listdir(directory)
    subdirectories = [item for item in all_contents]     
    for subject in subject_names: 
        for i in range(25): 
            name = subject + '-' + str(i) 
            if name not in subdirectories: 
                subdirectory_names_without_png.append(name) 

    print(subdirectory_names_without_png)
    return subdirectory_names_without_png


def find_directories_missing_folders(directory, folder_names):
    directories_missing_folders = []

    for subject in subject_names:
        for y in range(25):
            subdir_name = f"{subject}-{y}"
            subdir_path = os.path.join(directory, subdir_name)

            # Check if the directory exists
            if not os.path.exists(subdir_path):
                directories_missing_folders.append(subdir_name)
            else:
                tmp = os.listdir(subdir_path)
                if len(tmp) < len(folder_names):
                    directories_missing_folders.append(subdir_name)
    print(directories_missing_folders)
    return directories_missing_folders


if __name__ == "__main__":
    # directory = "./log_db"
    # subdirectory_names_without_png = find_subdirectory_names_without_png(directory)
    
    directory = './log_lora'
    folder_names = ['5', '6', '7', '8', '9']
    subdirectory_names_without_png = find_directories_missing_folders(directory, folder_names)

    indices = []
    print("Sub-directory names without PNG files:")
    for subdir_name in subdirectory_names_without_png:
        # Find the position in the subject_names list
        position = -1
        for idx, name in enumerate(subject_names):
            name = name + '-'
            if name in subdir_name:
                position = idx
                parts = subdir_name.split('-')
                num = int(parts[-1])
                index = idx * 25 + num
                indices.append(index)
                break
    list_str = ', '.join(map(str, indices))
    print(len(indices))
    print()
    print(list_str)

