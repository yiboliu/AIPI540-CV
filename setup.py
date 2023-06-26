import pip

# Installation is required for this project but it's not able to be installed everywhere except
pip.main(['install', 'git+https://github.com/facebookresearch/detectron2.git'])


# def download_image_files(bucket_name, destination_folder, prefix):
#     # Instantiate the client
#     client = storage.Client()
#
#     # Retrieve the bucket
#     bucket = client.get_bucket(bucket_name)
#
#     # List all blobs (files and subfolders) in the bucket
#     blobs = bucket.list_blobs(prefix=prefix)
#
#     # Iterate over the blobs and download image files
#     for blob in blobs:
#         if blob.name.endswith('.jpg'):
#             # Specify the destination path for the downloaded image file
#             destination_path = f"{destination_folder}/{blob.name}"
#
#             # Download the image file to the destination folder
#             blob.download_to_filename(destination_path)
#             print(f"Downloaded: {destination_path}")
#
#
# download_image_files('data-540-cv', 'train', 'train')
# download_image_files('data-540-cv', 'test', 'test')
