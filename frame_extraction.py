"""
This code will be the fram extraction for celeb-df-v2 dataset. It will:
- split frames into train/val/test using the official List_of_testing_videos.txt
- Extracts every 10th frame, then resizes to 224x224 and saves as .jpeg
- Lil neat thing is that it uses grab() to not blow up my macbook 
  to skip frames and not decode every frame. 
The libraries used are:
- cv2 for video processing, its apparently super useful for vision tasks.
- os for file handling and directory management.
- random for shuffling data.
- pathlib for path handling and making it more cross-platform.
- tqdm for progress bars so i can watch while doing other things.


playlist: Thief soundtrack tangerine dream
"""

#import libraries used for this 
import cv2, os, random
from pathlib import Path
from tqdm import tqdm

#configuring and setting up, the prep work if you will
#constants to easily work the paths when its smashing and dashing, figuratively
DATASET_ROOT = "/Volumes/Seagate/capstone/Celeb-DF-v2"
OUTPUT_ROOT = "/Volumes/Seagate/capstone/frames"

#the test list that specifies the testing videos
TEST_LIST = os.path.join(DATASET_ROOT, "List_of_testing_videos.txt")

#these constants are for the extraction process for the process
SAMPLE_RATE = 10 #10th frames
IMG_SIZE = 224 #resize to 224x224
JPEG_QUAL = 90 #quality for jpeg, 90 is pretty good, not too much compression
VAL_RATIO = 0.15 #15% ratio for vali
RANDOM_SEED = 301 #seed

#mapping the folders and their labels
FOLDERS = {
    "Celeb-real": "real",
    "YouTube-real": "real",
    "Celeb-synthesis": "fake"
}

#now down to business
def load_test_list(file_path):
    """
   Parses the official test list file for the splits.
    Args:
        file_path (str): Path to the test list file.
    Returns:
        test_videos (set): A set of video names that are in the test split.
    """

    test_videos = set()
    with open(file_path, 'r') as file:
        for line in file:
            video_name = line.strip()
            if not video_name:
                continue
            #format for. this is folder/filename.mp4
            parts = video_name.split(' ', 1) #split into folder and filename
            if len(parts) == 2: #check if split was successful
                test_videos.add(parts[1].strip()) #add the filename to the set
    #returns the set
    return test_videos

def extract_frames(video_path, output_dir, prefix):
    """
   Extracts every nth frame from a video, resizes it, and saves as JPEG.
   - Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        prefix (str): Prefix for the output frame filenames.
    - Returns:
        saved_count (int): The number of frames successfully extracted and saved.
    """

    cap = cv2.VideoCapture(str(video_path)) #open the video file using cv2, str() is used to convert from Path object to string path
    if not cap.isOpened(): #check if the video was opened successfully else error
        print(f"Error opening video file: {video_path}")
        return 0 #return 0 frames extracted if video can't be opened
    
    frame_count = 0 #init counter for frames
    saved_count = 0 #init counter for saved frames

    #using a while loop to read through the video frames idk how long it needs to be so no for loop
    while True:
        if frame_count % SAMPLE_RATE == 0: #check if the current frame is one we want to save based on the sample rate
            ret, frame = cap.read() #read the frame, ret is a boolean indicating if the read was successful, frame is the actual frame data
            if not ret: #if the read was not successful, we might have reached the end of the video or there was an error, so we
                break
            #resize the frame to the specified size and save it as a JPEG file with the specified quality
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) 
            file_name = f"{prefix}_frame_{frame_count:05d}.jpeg"
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUAL])
            saved_count += 1
        #if the frame is not one we want to save, we can use grab() to skip it without decoding, which is more efficient
        else:
            if not cap.grab():
                break
        frame_count += 1
        #cap.release() is called at the end of the function to free up resources, and the function returns the count of saved frames
    cap.release()
    return saved_count

def extract_pipeline():
    """
    This is the pipeline for extracting the frames from the videos in the dataset. It will:
    - Load the test list to determine which videos belong to the test split.
    - Iterate through the dataset folders and videos, determine their split (train/val/test)
      based on the test list, and extract frames accordingly.
    - For the training split, it will further split into train and validation sets based on the specified ratio.
    - Save the extracted frames in the appropriate directories with a consistent naming convention.
    - The function uses tqdm to show progress bars for the extraction process, making it easier to monitor.
    - The function also handles the creation of output directories if they do not exist, ensuring that the extracted frames are organized properly.

    void function
    """
    random.seed(RANDOM_SEED) #set seed

    #load the og test list set
    test_videos = load_test_list(TEST_LIST)
    #print so i can see the test videos loaded
    print(f"Loaded {len(test_videos)} test videos.")

    #output directories for the splits
    for split in ['train', 'val', 'test']: #iter through the splits
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok = True) #make the directories if they don't exist, exist_ok=True prevents error if they already exist

    #this dictionary will keep track of how many frames we save for each split and label, so we can print it at the end for sanity check
    total_saved = 0
    split_counts = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0}
    }

    for folder_name, label in FOLDERS.items(): #iter through the folders and their corresponding labels
        source_dir = os.path.join(DATASET_ROOT, folder_name) #get the source directory for the current folder
        videos = sorted([v for v in Path(source_dir).glob("*.mp4")
                 if not v.name.startswith("._")]) #get a sorted list of video files in the source directory, filtering out any hidden files that start with "._" which can be created by macOS and cause issues
        #print check for updates
        print(f"\nProcessing folder: {folder_name} with {len(videos)} videos.")

        #time to separate the test and non test videso 
        test_vids = []
        non_test_vids = []
        for video in videos:
            relative_path = f"{folder_name}/{video.name}"

            if relative_path in test_videos:
                test_vids.append(video)
            else:
                non_test_vids.append(video)

        #splitting non test into train and val
        random.shuffle(non_test_vids) #shuffle the non test videos to randomize the train/val split
        val_count = int(len(non_test_vids) * VAL_RATIO) #calculate how many videos should go into the validation set based on the specified ratio
        val_vids = non_test_vids[:val_count] #first part goes to validation
        train_vids = non_test_vids[val_count:] #the rest goes to training

        print(f" -> train: {len(train_vids)} videos, val: {len(val_vids)} videos, test: {len(test_vids)} videos.")

        #now we have the splits so its extraction time
        for split_name, vid_list in [("train", train_vids),
                                    ("val", val_vids),
                                    ("test", test_vids)]:
            output_dir = os.path.join(OUTPUT_ROOT, split_name, label) #get the output directory for the current split and label

            for video in tqdm(vid_list, desc=f"{folder_name} [{split_name}]"):
                prefix = f"{folder_name}_{video.stem}"
                saved = extract_frames(video, output_dir, prefix) #extract the frames from the video and save them to the output directory with the specified prefix
                total_saved += saved #update the total saved count
                split_counts[split_name][label] += saved #update the count for the current split

    #printing a summary for I when the extraction is done to see success or failure and revidse
    print(f"\nExtraction complete. Total frames saved: {total_saved}\n")


    for split in ["train", "val", "test"]:
        real_count = split_counts[split]["real"]
        fake_count = split_counts[split]["fake"]
        total_count = real_count + fake_count
        ratio = fake_count / real_count if real_count > 0 else 0
        print(f"{split.capitalize()} - Real: {real_count}, Fake: {fake_count}, Total: {total_count}, Fake/Real Ratio: {ratio:.2f}")

        #checkinh the disk usage also for sanity check and to see how my mac handled the stress of extraction
        total_bytes = 0
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                folder = os.path.join(OUTPUT_ROOT, split, label)
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        total_bytes += os.path.getsize(os.path.join(folder, file))
        print(f"\n Total size: {total_bytes / (1024 * 1024 * 1024):.2f} GB")

if __name__ == "__main__":
    extract_pipeline()