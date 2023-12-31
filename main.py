import os
import streamlit as st
import model_serving

if __name__ == "__main__":
    dest = 'user_pics'  # this directory is git ignored, contains only original and processed images.
    if not os.path.exists(dest):
        os.mkdir(dest)

    upload = st.file_uploader("Upload a file")
    if upload:
        assert(upload.name.endswith('jpg'), 'The file must be a jpg file')
        new_img_name = f"{upload.name.split('.')[0]}-with-keypoints.jpg"
        if st.button('Save File'):
            # Save the file to a designated place so that we can read later
            with open(os.path.join(dest, upload.name), "wb") as f:
                f.write(upload.getbuffer())
            # Predict wheat key points in the uploaded picture
            model_serving.predict_img_svm(
                img_path=os.path.join(dest, upload.name),
                new_img_name=new_img_name,
                dest_folder=dest
            )
            # Show the picture in the UI
            st.image(os.path.join(dest, new_img_name))
