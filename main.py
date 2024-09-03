import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_navigation_bar import st_navbar
#Creating a Tensorflow model for prediction
def modelpredictions(test_image):
    model = tf.keras.models.load_model("Trained_model.keras")
    # this is the image preprocessing part
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])#convert single image to a batch
    prediction = model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

#ui of the app
st.sidebar.title('Dashboard')
appmode = st.sidebar.selectbox('Select Page',['Home','About','Disease Detection','Contact us'])


# navbar = st_navbar(['Home','About','Disease Detection','Contact Us'])
# st.write(navbar)

# Home Page
if appmode=='Home':
    st.header('AI BASED CROP DISEASE DETECTION SYSTEM')
    image_path='home_image_2.jpg'
    st.image(image_path)
    st.markdown('''
    The AI-Based Crop Disease Detection project is designed to assist farmers and agricultural professionals in identifying crop diseases through a user-friendly interface. By leveraging advanced image processing and machine learning techniques, the software analyzes uploaded images of crops and provides accurate predictions regarding potential diseases.

    ### Features
                    
       1) **User-Friendly Interface**: The application features a simple and intuitive interface that allows users to upload images easily without requiring technical expertise.
       2) **Image Analysis**: The software employs state-of-the-art AI algorithms to analyze the uploaded images, identifying patterns and anomalies that may indicate disease.
       3) **Disease Prediction**: After analyzing the image, the system generates a prediction of the crop disease present, along with relevant information about the disease and possible remedies.
       4) **Fast Processing**: The model is optimized for quick image processing, ensuring that users receive timely feedback on their crop health.

    ### How It Works
                    
       1) **Image Upload**: Users can upload images of their crops directly through the interface. The system supports various image formats for convenience.
                
       2) **Image Processing**: Once an image is uploaded, the software preprocesses the image to enhance quality and prepare it for analysis.
                
       3) **Disease Detection**: The AI model analyzes the image using trained algorithms to detect signs of diseases. It compares the input image against a database of known crop diseases.
                
       4) **Output Generation**: The system generates a report that includes:
         The predicted disease
         Confidence level of the prediction
         Recommended actions or treatments
                

    ### System Requirements
                
       1) **Hardware**: A computer or device with a minimum of 4 GB RAM and a modern processor.
       2) **Software**: Compatible with Windows, macOS, or Linux operating systems. Requires a web browser for the interface.
                
    ### Getting Started
                
       1) **Installation**: Follow the installation instructions provided in the setup guide to install the software on your device.
       2) **Launching the Application**: Open the application and navigate to the upload section.
       3) **Uploading Images**: Click on the upload button, select an image of the crop, and submit it for analysis.
       4) **Interpreting Results**: After processing, review the predicted disease and suggested actions.
                
    ### Use Cases
                
       1) **Farmers**: Quickly identify crop diseases to take timely action and minimize crop loss.
       2) **Agricultural Advisors**: Provide expert advice based on disease predictions to farmers.
       3) **Research Institutions**: Analyze disease patterns and improve crop management strategies.


    ''') 

elif appmode=='About':
    st.header('About Us')
    
    