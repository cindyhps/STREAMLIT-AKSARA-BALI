import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import cv2
import numpy as np
from skimage.feature import hog

def data_prep(canvas_result): 
    canvas_image = canvas_result.image_data
    
    # Konversi ke dalam format array NumPy
    canvas_array = np.array(canvas_image)

    # Resize gambar menjadi 32x32
    resized_image = cv2.resize(canvas_array, (25, 25))
    
    # Konversi gambar menjadi grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    
    # Lakukan proses thresholding
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary_image
    
def extract_feature(image): 
    px = 5
    
    # Hitung HOG descriptor
    features = hog(image, orientations=9, pixels_per_cell=(px, px),
                   cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    
    return features
    

def predict_aksara(canvas): 
    model = joblib.load('svm_poly.pkl')
    
    # Preprocessing data
    preprocessed_data = data_prep(canvas)
    
    # Ekstraksi fitur
    features = extract_feature(preprocessed_data)
    
    # Lakukan prediksi dengan model
    prediction = model.predict(features.reshape(1, -1))
    
    return prediction

st.write("""
# Deteksi Aksara Bali
""")

st.write("Tuliskan aksara dan kami akan memprediksi nama dari aksara tersebut dan menampilkan informasi tentang aksara tersebut")

stroke_width = 10
stroke_color = "#eee"
bg_color = "#000000"

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

nama_aksara = [
    'Carik', #0
    'Carik kalih', #1
    'taling', #2
    'panti' #3
    'cecek', #4
    'pamada', #5
    'pamada', #6
    '0', #7
    '1', #8
    '2', #9
    '3', #10
    '4', #11
    '5', #12
    '6', #13
    '7', #14
    '8', #15
    '9', #16
    'a-kara', #17
    'a_kara tedong', #18
    'i-kara', #19
    'i-kara tedong', #20
    'u-kara', #21
    'u-kara tedong', #22
    'e-kara', #23
    'je-jera', #24
    'o-kara', #25
    'o-kara tedong', #26
    'Ha', #27
    'Na', #28
    'Ca', #29
    'Ra', #30
    'Ka', #31
    'Da', #32
    'Ta', #33
    'Sa', #34
    'Wa', #35
    'La', #36
    'Ma', #37
    'Ga', #38
    'Ba', #39
    'Nga', #40
    'Pa', #41
    'Ja', #42
    'Ya', #43
    'Nya', #44
    'Na Rambat', #45
    'Da Madu', #46
    'Ta Tawa', #47
    'Ta Latik', #48
    'Sa Saga', #49
    'Sa Sapa', #50
    'Ga Gora', #51
    'Ba Kembang', #52
    'Pa Kapal', #53
    'Ca Laca', #54
    'Kha', #55
    'Taleng', #56
    'Ulu', #57
    'Ulu Sari', #58
    'Suku', #59
    'Suku Ilut', #60
    'Taleng', #61
    'Taleng Marepa', #62
    'Taleng Tedong', #63
    'Taleng Marepa Tedong', #64
    'Pepet', #65
    'Pepet Tedong', #66
    'Ulu Candra', #67
    'Ulu Ricem', #68
    'Cecek', #69
    'Surang', #70
    'Bisah', #71
    'Adeg-adeg', #72
    'Gantungan A/Ha', #73
    'Gantungan Na', #74
    'Gantungan Ca', #75
    'Gantungan Ra', #76
    'Gantungan Ka', #77
    'Gantungan Da', #78
    'Gantungan Ta', #79
    'Gempelan Sa', #80
    'Gantungan Wa', #81
    'Gantungan La', #82
    'Gantungan Ma', #83
    'Gantungan Ga', #84
    'Gantungan Ba', #85
    'Gantungan Nga', #86
    'Gempelan Pa', #87
    'Gantungan Ja', #88
    'Gantungan Ya', #89
    'Gantungan Nya', #90
    'Gantungan Na Rambat', #91
    'Gantungan Da Madu', #92
    'Guung Mecelek', #93
    'Gantungan Ta Latik', #94
    'Gantungan Sa Saga', #95
    'Gempelan Sa Sapa', #96
    'Ga Gora', #97
    'Gantunga Ba Kembang', #98
    'Gantungan Ta Latik', #99
    'Gantungan Ca Laca', #100
    'Gantungan Kha', #101
]

# Buat tombol untuk melakukan prediksi
if st.button("Prediksi"):
    prediction = predict_aksara(canvas_result)
    index_pred = int(prediction[0])
    st.write(f"**Prediksi Aksara:** {index_pred}")
