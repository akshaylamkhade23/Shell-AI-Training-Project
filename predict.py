import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import google.generativeai as genai

st.set_page_config(
    page_title="ESMP-HomePage",
    page_icon="🌿",
)
#st.title("🍃Eco-Sustainability with Medicinal Plants")
st.markdown("""
    <h1 style="font-family: 'Courier New', monospace; font-size: 50px; color:#32CD32;">
            🍃Eco-Sustainability with Medicinal Plants
    </h1>
""", unsafe_allow_html=True)
st.sidebar.image("data:image/jpeg;base64,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",width=150)
st.sidebar.title("🌿ESMP")
st.sidebar.markdown("""
    ESMP uses deep learning to identify and classify medicinal plant species from imagery and monitor their population changes over time. 
    
    -->Page1
    
    --->page2
""")
st.sidebar.success("Select a page above")
st.sidebar.markdown("""
    This is a beautiful Streamlit app UI with Lottie animations, charts, and interactive elements.
""")


# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('leaf model.h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (256, 256)  # Same size used during training

def preprocess_image(image):
    img = image.resize(IMG_SIZE)  # Resize image
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App UI
#st.title("Plant Health Classifier")
st.markdown("""
    <h1 style="font-family: 'Roboto', sans-serif; font-size: 45px; color:#FFD700;">
        Monitoring and Prediction of Medicinal Plant
    </h1>
""", unsafe_allow_html=True)
st.subheader("Classify images into **Early**, **Healthy**, or **Late** stages.")
st.markdown("""
    <style>
        /* Global style */
        .main {
            background-color: #F1F5F9;
            font-family: 'Helvetica Neue', sans-serif;
            color: #333;
            padding: 30px;
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #4F46E5;
            color: white;
            padding: 20px;
        }
        .sidebar.sidebar-image{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            
        }

        /* Title and header styles */
        .header {
            font-size: 36px;
            color: #4F46E5;
            font-weight: 600;
        }

        .subheader {
            font-size: 24px;
            color: #1E293B;
            font-weight: 500;
            margin-top: 20px;
        }

        /* Styled buttons */
        .stButton > button {
            background-color: #4F46E5;
            color: white;
            border-radius: 5px;
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #3B3BE3;
        }

        /* Text styling */
        .stMarkdown {
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 20px;
        }

        .footer {
            text-align: center;
            padding: 20px;
            font-size: 14px;
            color: #C0C0C0;
            margin-top: 40px;
        }
        
        /* Footer link styling */
        .footer a {
            color:#FFD700;
            text-decoration: blink;
        }

        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)


# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            # Preprocess and predict
            image = load_img(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            # Class labels
            class_labels = [
                "Aloe Vera", "Amla", "Amruthaballi", "Arali", "Astma_weed", "Badipala","Balloon_Vine", "Bamboo", "Beans","Betel", 
                "Bhrami", "Bringaraja", "Caricature", "Castor", "Catharanthus", "Chakte", "Chilly","Citron lime (herelikai)", 
                "Coffee", "Commonrue(naagdalli)", "Coriender", "Curry", "Doddpathre","Drumstick", "Ekka", "Eucalyptus", "Ganigale", 
                "Ganike", "Gasagase", "Ginger", "Globe Amarnath","Guava", "Henna", "Hibiscus", "Honge", "Insulin", "Jackfruit", 
                "Jasmine", "Kambajala", "Kasambruga", "Kohlrabi", "Lantana", "Lemon", "Lemongrass", "Malabar_Nut","Malabar_Spinach", 
                "Mango", "Marigold","Mint", "Neem", "Nelavembu", "Nerale", "Nooni", "Onion", "Padri", "Palak(Spinach)", "Papaya", 
                "Parijatha", "Pea", "Pepper", "Pomoegranate", "Pumpkin", "Raddish", "Rose", "Sampige", "Sapota","Seethaashoka", 
                "Seethapala", "Spinach1", "Tamarind", "Taro", "Tecoma", "Thumbe", "Tomato", "Tulsi","Turmeric", "ashoka", "camphor", 
                "kamakasturi", "kepala"
            ]

            # Get predicted class
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class_index]

            # Display prediction
            st.success(f"🌿 Predicted Class: *{predicted_label}* ")

            # Set up Gemini for plant info retrieval
            genai.configure(api_key="AIzaSyC2RMwRXWiRgaR1s46oLYIlD4Ygrejktoo")

            # Use Gemini to generate a response
            model = genai.GenerativeModel("gemini-pro")

            def get_plant_info(plant_name):
                search_query = f"{plant_name} medicinal plant"

                # Requesting the summary, current quantity, and area of availability
                response = model.generate_content(f"Provide a summary about the medicinal plant {search_query}. Include its benefits, uses, current cultivation in acres in Maharashtra and areas of availability with cultivation in acres in districts of Maharashtra India.")

                if response and response.text:
                    return {
                        "Title": plant_name.title(),
                        "Summary": response.text, 
                    }
                else:
                    return {"Error": f"No relevant information found for {plant_name}"}

            # Fetching plant info
            plant_name = class_labels[predicted_class_index]
            plant_info = get_plant_info(plant_name)

            # Display plant information
            st.markdown(f"## {plant_info.get('Title', 'N/A')}")
            st.markdown(plant_info.get('Summary', 'No summary available'))

            if 'Error' in plant_info:
                st.error(plant_info['Error'])

            # Assuming `model.generate_content` now includes the current quantity and area of availability,
            # Let's display those values in a structured format

            # For demonstration purposes, let's assume the response text includes these details in specific sections:
            
                    
                
            
st.markdown("""
    <div class="footer">
        Created by @akshaylamkhade</a> 
    </div>
""", unsafe_allow_html=True)