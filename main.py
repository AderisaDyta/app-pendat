import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 

st.write(""" 
# Aplikasi Heart Disease Dataset 
By. Aderisa Dyta Okvianti(200411100013)
""")

st.write("----------------------------------------------------------------------------------")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Dataset", "Preprocessing", "Modeling", "Implementation"])

with tab1:
    st.subheader("""Pengertian""")
    st.write("""
    Dataset ini digunakan untuk menentukan apakah pasien menderita penyakit Hati atau tidak.
    """)

    st.markdown(
        """
        Dataset ini memiliki beberapa fitur yaitu :
       - Male : LAKI (p=0,L=1)
       - Age: Umur
       - education: Pendidikan
       - CurrentSmoker: perokok saat ini (0:tidak merokok, 1=merokok)
       - cigsPerDay: Jumlah rokok yang dikonsumsi per hari
       - BPMeds: Status seseorang mengomsumsi obat tekanan darah (0:tdk konsumsi, 1:konsumsi)
       - PrevalanseStroke: angka kejadian stroke (0:tdk, 1:mengalami)
       - LazimHyp: angka kejadian hipertensi (0:tdk mengalami hipertensi, 1:mengalami hipertensi)
       - diabetes: status diabetes atau tidak (0:tdk, 1:iya)
       - totChol:tingkat kolesterol(mm/dl)
        """
    )

    st.subheader("""Dataset""")
    st.write("""
    Dataset penyakit Hati ini diambil dari Kaggle 
    <a href="https://www.kaggle.com/datasets/muhammadroyyanrozani/memprediksi-penyakit-hati">Dataset</a>""", unsafe_allow_html=True)

with tab2:
    st.subheader("""Heart Disease Dataset""")
    df = pd.read_csv('https://raw.githubusercontent.com/AderisaDyta/dataminingnew/main/framingham.csv')
    df = df.dropna(axis = 0)
    st.dataframe(df) 

with tab3:
    st.subheader("""Rumus Normalisasi Data""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['TenYearCHD','education'])
    y = df['TenYearCHD'].values
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.TenYearCHD).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Positive' : [dumies[1]],
        'Negative' : [dumies[0]]
    })

    st.write(labels)

with tab4:
    st.subheader("""Metode Yang Digunakan""")
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

    with st.form("modeling"):
        st.write("Pilih Metode yang digunakan : ")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-NN')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier(random_state=1)
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("K-NN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik Akurasi")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

with tab5:
        st.subheader("Form Implementasi")
        with st.form("my_form"):
            male = st.slider('Jenis Kelamin Pasien', 0,1)
            age = st.number_input('Usia pasien',0)
            currentSmoker = st.slider('perokok saat ini',0,1)
            cigsPerDay = st.number_input('Jumlah konsumsi rokok pasien per hari',0)
            BPMeds = st.number_input('Status seseorang mengomsumsi obat tekanan darah')
            PrevalanseStroke = st.slider('angka kejadian stroke',0,1)
            LazimHyp = st.slider('angka kejadian hipertensi',0,1)
            diabetes = st.slider('status diabetes atau tidak',0,1)
            totChol = st.number_input('tingkat kolesterol')
            sysBP = st.number_input('sysBP')
            diaBP = st.number_input('diaBP')
            BMI = st.number_input('BMI')
            heartRate = st.number_input('heartRate')
            glucose = st.number_input('glucose')
            model = st.selectbox('Model untuk prediksi',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    male,
                    age,
                    currentSmoker,
                    cigsPerDay,
                    BPMeds,
                    PrevalanseStroke,
                    LazimHyp,
                    diabetes,
                    totChol,
                    sysBP,
                    diaBP,
                    BMI,
                    heartRate,
                    glucose

                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                if model == 'K-NN':
                    mod = knn 
                if model == 'Decision Tree':
                    mod = dt
                
                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')