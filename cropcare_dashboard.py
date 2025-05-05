import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
import requests
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- GLOBAL SETUP ----------------
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ---------------- HEADER IMAGE ----------------
header_image_path = "sample_gallery/tomato_leaf_mold.jpg"
if os.path.exists(header_image_path):
    header_image = Image.open(header_image_path)
    st.image(header_image, use_column_width=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = r"C:\Users\vishal\Downloads\CNN_plantdiseases_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- CLASS NAMES ----------------
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ---------------- MODEL PREDICTION FUNCTION ----------------
def model_predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    prediction_probs = model.predict(img)[0]
    index = np.argmax(prediction_probs)
    confidence = prediction_probs[index] * 100
    return class_names[index], confidence

# ---------------- WEATHER FUNCTION ----------------
def show_weather():
    st.subheader("🌤 Real-Time Weather")
    city = st.text_input("Enter your city:", "Delhi")
    if city:
        url = f"https://wttr.in/{city}?format=%l:+%c+%t\nHumidity:+%h\nWind:+%w\nCondition:+%C"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                st.success(f"📍 {lines[0]}")
                for line in lines[1:]:
                    if "Humidity" in line:
                        st.info(f"💧 {line}")
                    elif "Wind" in line:
                        st.info(f"🌬 {line}")
                    elif "Condition" in line:
                        condition = line.split(":")[-1].strip().lower()
                        st.info(f"📋 Condition: {condition.capitalize()}")
                        if "rain" in condition:
                            st.warning("🌧 Rain alert. Protect your plants.")
                        elif "sun" in condition:
                            st.success("☀ Sunny! Good for photosynthesis.")
                        elif "cloud" in condition:
                            st.info("☁ Cloudy. Check humidity.")
                        elif "storm" in condition:
                            st.error("🌩 Storm! Secure your crops.")
                        elif "fog" in condition:
                            st.info("🌫 Foggy. Avoid overwatering.")
            else:
                st.error("⚠ Weather service unavailable.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🌿 Plant Disease Detection")
app_mode = st.sidebar.selectbox("Navigate", [
    "🏠 Home", "🔍 Disease Recognition", "🖼 Gallery", "📊 Model Performance", "ℹ About", "📞 Contact", "❓ Help", "📜 History", "📝 Feedback"
])

# ---------------- HOME PAGE ----------------
def show_home():
    st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🛠 How It Works")
    st.markdown("""
    1. 📷 Upload a leaf photo  
    2. 🤖 AI analyzes it  
    3. 🩺 Get predicted disease + confidence  
    4. 🚜 Act to protect crops  
    """)
    st.subheader("✨ Features")
    st.markdown("""
    - Detects 38+ plant diseases  
    - Fast, accurate AI model  
    - Weather awareness for better decisions  
    - Clean, user-friendly interface  
    """)
    show_weather()

# ---------------- DISEASE DETECTION ----------------
def show_disease_recognition():
    st.title("🔍 Disease Recognition")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Predict"):
            with st.spinner("Analyzing..."):
                prediction, confidence = model_predict(file_path)
            st.success(f"🌾 Predicted Disease: **{prediction}**")
            st.info(f"📊 Confidence: {confidence:.2f}%")
            show_disease_info(prediction)
            show_treatment_tips(prediction)

def show_model_performance():
    st.title("📊 Model Performance Metrics")

    # Load or define your true and predicted labels
    try:
        df = pd.read_csv("predictions.csv")  # Ensure this CSV exists
        y_true = df['true'].tolist()
        y_pred = df['pred'].tolist()
    except Exception as e:
        st.error("❌ Could not load predictions.csv. Ensure the file exists and contains 'true' and 'pred' columns.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    support = [report[cls]['support'] for cls in class_names]
    per_class_accuracy = [cm[i][i] / support[i] if support[i] > 0 else 0 for i in range(len(class_names))]

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

    # Accuracy and Support
    st.subheader("📊 Per-Class Accuracy & Support")
    fig_bars, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].bar(class_names, per_class_accuracy, color='skyblue')
    axs[0].set_ylim(0, 1.1)
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Per-Class Accuracy")
    axs[0].tick_params(axis='x', rotation=90)

    axs[1].bar(class_names, support, color='lightcoral')
    axs[1].set_ylabel("Samples")
    axs[1].set_title("Support per Class")
    axs[1].tick_params(axis='x', rotation=90)
    st.pyplot(fig_bars)

    st.subheader("📋 Full Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

# ---------------- DISEASE INFO ----------------
def show_disease_info(disease_name):
    st.subheader(f"🦠 Disease Info: {disease_name}")
    info = {
    'Apple__Apple_scab': "Fungal disease caused by *Venturia inaequalis*. Appears as dark, velvety spots on leaves and fruit. Causes premature leaf drop.",
    'Apple_Black_rot': "Caused by *Botryosphaeria obtusa*. Results in fruit rot, leaf spots, and branch cankers.",
    'Apple_Cedar_apple_rust': "Fungal infection that alternates between apple and cedar trees. Produces bright orange spots on leaves.",
    'Apple__healthy': "Leaves, stems, and fruits show no signs of infection. Tree is in optimal health.",
    'Blueberry__healthy': "No visible signs of disease or nutrient deficiency. Healthy growth.",
    'Cherry(including_sour)Powdery_mildew': "Caused by *Podosphaera clandestina*. White powdery growth on leaf surfaces; affects fruit set.",
    'Cherry(including_sour)_healthy': "Normal leaf coloration and growth. No fungal or bacterial symptoms.",
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot': "Caused by *Cercospora zeae-maydis*. Gray lesions on leaves; reduces photosynthetic area.",
    'Corn(maize)Common_rust': "Red to brown pustules on leaves. Caused by *Puccinia sorghi* fungus.",
    'Corn_(maize)Northern_Leaf_Blight': "Elongated grayish-green lesions caused by *Exserohilum turcicum*. Major foliar disease in maize.",
    'Corn(maize)_healthy': "No signs of blight or rust. Leaves are uniform green.",
    'Grape__Black_rot': "Fungal disease by *Guignardia bidwellii*. Dark spots on leaves and fruit rot on berries.",
    'Grape__Esca(Black_Measles)': "Complex disease caused by fungi like *Phaeomoniella*. Results in tiger-striped leaves and vine decline.",
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': "Spots on leaves, may cause them to drop. Caused by *Pseudocercospora vitis*.",
    'Grape___healthy': "Vine growth is healthy. Leaves and fruit are disease-free.",
    'Orange__Haunglongbing(Citrus_greening)': "Serious bacterial disease spread by psyllid insects. Causes yellowing and bitter, misshapen fruit.",
    'Peach__Bacterial_spot': "Caused by *Xanthomonas campestris*. Water-soaked lesions on fruit and leaves.",
    'Peach__healthy': "Healthy foliage and fruit development with no dark or wet spots.",
    'Pepper,bell_Bacterial_spot': "Small, dark, water-soaked lesions. Leads to defoliation and yield loss.",
    'Pepper,_bell_healthy': "Bright green leaves and healthy fruit set. No visible lesions.",
    'Potato__Early_blight': "Fungal disease with concentric ring patterns on older leaves. Caused by *Alternaria solani*.",
    'Potato__Late_blight': "Aggressive disease caused by *Phytophthora infestans*. Brown-black lesions with white edges.",
    'Potato_healthy': "No signs of yellowing, spots or curling. Optimal plant health.",
    'Raspberry_healthy': "Vigorous shoot and fruit development. No curling or discoloration.",
    'Soybean__healthy': "Uniform green foliage. No leaf spots or growth stunting.",
    'Squash__Powdery_mildew': "White, powdery fungal growth on leaf surface. Reduces photosynthesis.",
    'Strawberry_Leaf_scorch': "Caused by *Diplocarpon earlianum*. Reddish-brown spots, leaf edge burning.",
    'Strawberry__healthy': "Green, shiny leaves with no necrotic patches.",
    'Tomato__Bacterial_spot': "Caused by *Xanthomonas*. Dark, greasy lesions on leaves and fruit.",
    'Tomato_Early_blight': "Caused by *Alternaria*. Target-like spots on older leaves.",
    'Tomato__Late_blight': "Highly destructive. Water-soaked lesions; fruits rot quickly.",
    'Tomato__Leaf_Mold': "Fungal disease. Yellowing on top leaf surface and mold on bottom.",
    'Tomato_Septoria_leaf_spot': "Small, circular spots with gray centers. Caused by *Septoria lycopersici*.",
    'Tomato__Spider_mites Two-spotted_spider_mite': "Tiny pests causing yellow stippling. Severe infestations lead to webbing.",
    'Tomato__Target_Spot': "Dark brown spots with concentric rings. May coalesce and kill tissue.",
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': "Transmitted by whiteflies. Causes upward curling of yellow leaves.",
    'Tomato__Tomato_mosaic_virus': "Causes mottled, malformed leaves. Spread by contact or contaminated tools.",
    'Tomato___healthy': "Green, robust leaves and fruit. No curling or spots."
}

    st.write(info.get(disease_name, "No specific information available."))

# ---------------- TREATMENT TIPS ----------------
def show_treatment_tips(disease_name):
    st.subheader("💊 Suggested Treatment")
    tips = {
    'Apple__Apple_scab': "Apply fungicides like Captan. Remove and destroy fallen leaves.",
    'Apple_Black_rot': "Prune infected twigs. Apply Mancozeb during bloom.",
    'Apple_Cedar_apple_rust': "Use resistant varieties. Apply fungicides early in the season.",
    'Apple__healthy': "Maintain general care. No treatment needed.",
    'Blueberry__healthy': "Keep soil acidic. Regular pruning for airflow.",
    'Cherry(including_sour)Powdery_mildew': "Apply sulfur-based fungicides. Ensure good spacing.",
    'Cherry(including_sour)_healthy': "Maintain balanced fertilizer and water schedule.",
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot': "Use resistant hybrids. Apply fungicides at tassel stage.",
    'Corn(maize)Common_rust': "Resistant varieties are preferred. Use fungicides for severe cases.",
    'Corn_(maize)Northern_Leaf_Blight': "Plant resistant varieties. Use fungicide if needed.",
    'Corn(maize)_healthy': "Maintain nitrogen levels. Ensure irrigation during flowering.",
    'Grape__Black_rot': "Remove infected grapes. Apply early-season fungicides.",
    'Grape__Esca(Black_Measles)': "Avoid pruning wounds. Remove infected wood.",
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': "Spray with copper fungicides. Improve air circulation.",
    'Grape___healthy': "Regular trimming and proper irrigation is sufficient.",
    'Orange__Haunglongbing(Citrus_greening)': "Remove infected trees. Control psyllid vectors.",
    'Peach__Bacterial_spot': "Use resistant cultivars. Apply copper sprays during bud swell.",
    'Peach__healthy': "Keep orchard clean. Avoid excess nitrogen.",
    'Pepper,bell_Bacterial_spot': "Use certified seeds. Apply copper-based bactericides.",
    'Pepper,_bell_healthy': "Avoid overhead watering. Monitor regularly.",
    'Potato__Early_blight': "Use chlorothalonil or mancozeb. Rotate crops annually.",
    'Potato__Late_blight': "Apply systemic fungicides. Remove infected debris.",
    'Potato_healthy': "Use certified seeds. Avoid waterlogging.",
    'Raspberry_healthy': "Ensure adequate air circulation. Prune old canes.",
    'Soybean__healthy': "Rotate with non-legumes. Monitor nitrogen needs.",
    'Squash__Powdery_mildew': "Apply neem oil or bicarbonate sprays. Improve spacing.",
    'Strawberry_Leaf_scorch': "Remove infected leaves. Use fungicides like Captan.",
    'Strawberry__healthy': "Use straw mulch. Remove runners as needed.",
    'Tomato__Bacterial_spot': "Avoid overhead watering. Copper sprays help early.",
    'Tomato_Early_blight': "Use mulch and copper fungicides. Stake plants for airflow.",
    'Tomato__Late_blight': "Apply metalaxyl or chlorothalonil. Destroy infected plants.",
    'Tomato__Leaf_Mold': "Use copper-based sprays. Improve ventilation.",
    'Tomato_Septoria_leaf_spot': "Remove affected leaves. Spray with chlorothalonil.",
    'Tomato__Spider_mites Two-spotted_spider_mite': "Spray with miticides. Use insecticidal soap.",
    'Tomato__Target_Spot': "Use crop rotation. Apply mancozeb-based fungicides.",
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies. Remove infected plants.",
    'Tomato__Tomato_mosaic_virus': "Disinfect tools. Avoid handling plants when wet.",
    'Tomato___healthy': "Maintain fertilization. Water consistently."
}

    st.info(tips.get(disease_name, "Treatment info will be updated soon."))

# ---------------- GALLERY ----------------
def show_gallery():
    st.title("🖼 Leaf Gallery")
    gallery_folder = "sample_gallery"
    image_captions = {
        "apple_healthy.jpg": "Apple – Healthy",
        "corn_blight.jpg": "Corn – Leaf Blight",
        "tomato_leaf_mold.jpg": "Tomato – Leaf Mold"
    }
    for img_file, caption in image_captions.items():
        img_path = os.path.join(gallery_folder, img_file)
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_column_width=True)

# ---------------- DISEASE DETECTION HISTORY ----------------
def show_history():
    st.title("📜 Detection History")
    if 'history' not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        for idx, record in enumerate(st.session_state.history[:5]):
            st.write(f"{idx+1}. **Disease**: {record['disease']}, **Confidence**: {record['confidence']}%")
    else:
        st.write("No history available.")

# ---------------- FEEDBACK ----------------
def show_feedback():
    st.title("📝 Feedback")
    feedback = st.text_area("Please provide your feedback or suggestions:")
    if st.button("Submit"):
        if feedback:
            st.success("Thank you for your feedback!")
            # Here, you can save the feedback to a file or database
        else:
            st.warning("Please provide your feedback before submitting.")

# ---------------- ROUTER ----------------
if app_mode == "🏠 Home":
    show_home()
elif app_mode == "🔍 Disease Recognition":
    show_disease_recognition()
elif app_mode == "🖼 Gallery":
    show_gallery()
elif app_mode == "📊 Model Performance":
    show_model_performance()
elif app_mode == "📜 History":
    show_history()
elif app_mode == "📝 Feedback":
    show_feedback()
elif app_mode == "ℹ About":
    st.title("ℹ About")
    st.write("This project is a deep learning-based plant disease detector built using TensorFlow and Streamlit.")
elif app_mode == "📞 Contact":
    st.title("📞 Contact")
    st.write("For queries or feedback, contact us at: support@plantcare.ai")
elif app_mode == "❓ Help":
    st.title("❓ Help")
    st.markdown("""
    - Upload high-resolution images with a clear leaf focus  
    - Make sure leaves are not blurry or shadowed  
    - For best results, upload one image at a time  
    """)
