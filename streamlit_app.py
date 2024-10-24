import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import ast
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(layout="wide", page_title="電池配方暨電量趨勢分析工具")

# 分頁選擇
st.sidebar.title("Battery Model App")
page = st.sidebar.radio("選擇頁面", ["訓練模型", "試試手"])

# 分頁 1: 訓練模型
if page == "訓練模型":
    st.title("訓練回歸模型")

    # 1. 上傳資料
    csv_uploaded_file = st.file_uploader("上傳訓練資料 (CSV)", type=["csv"])
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    if csv_uploaded_file is not None:
        data = pd.read_csv(csv_uploaded_file)
        data['battery_capacity'] = data['battery_capacity'].apply(lambda x: np.array(ast.literal_eval(x)))

        options = st.multiselect(
            "預測模型種類(圈數)",
            ["600", "800", "1000", "1600", "3000"],
            ["600"],
        )

        if st.button("開始訓練模型"):
            progress_text = "模型訓練中，請稍候..."
            my_bar = st.progress(0, text=progress_text)

            # Create DataFrame without transposing
            df = pd.DataFrame(data['battery_capacity'].tolist())

            for percent_complete in range(len(options)):

                pred_circle_no = int(options[percent_complete])

                X = df.iloc[:, :pred_circle_no-2]
                y = df.iloc[:, pred_circle_no-1]

                # Split the data into training and testing sets (80% train, 20% test)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                # Train the regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions on training and test sets
                train_prediction = model.predict(X_train)
                test_prediction = model.predict(X_test)

                # Calculate percentage error for training and test sets
                train_error = root_mean_squared_error(y_train, train_prediction)
                test_error = root_mean_squared_error(y_test, test_prediction)

                # Calculate and display the average percentage error
                train_result = "小於1%，誤差極小" if train_error < 0.01 else train_error
                test_result = "小於1%，誤差極小" if test_error < 0.01 else test_error

                st.write(f"Circle No. {pred_circle_no}: 訓練集預測誤差為【{train_result}】")
                st.write(f"Circle No. {pred_circle_no}: 測試集預測誤差為【{test_result}】")

                # Save the model to a file
                model_name = f'lr_model_{pred_circle_no}.pkl'
                with open(os.path.join("models", model_name), 'wb') as file:
                    pickle.dump(model, file)

                # st.write(f"→   {model_name.split('.')[0]} 模型訓練完成！")

                # Create subplots with 1 row and 2 columns
                fig = make_subplots(rows=1, cols=2, subplot_titles=("訓練集", "測試集"))

                # Add actual vs. predicted line for training data in the first subplot (column 1)
                fig.add_trace(go.Scatter(x=np.arange(len(y_train)), y=y_train, mode='lines', name='訓練集實際值', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(len(train_prediction)), y=train_prediction, mode='lines', name='訓練集預測值', line=dict(dash='dash', color='orange')), row=1, col=1)

                # Add actual vs. predicted line for test data in the second subplot (column 2)
                fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='測試集實際值', line=dict(color='green')), row=1, col=2)
                fig.add_trace(go.Scatter(x=np.arange(len(test_prediction)), y=test_prediction, mode='lines', name='測試集預測值', line=dict(dash='dash', color='red')), row=1, col=2)

                # Update layout for better readability
                fig.update_layout(
                    title=f"Circle No. {pred_circle_no} 訓練與測試集預測結果",
                    xaxis_title="資料索引",
                    yaxis_title="數值",
                    legend_title="資料類型",
                    template="plotly_white",
                    showlegend=True
                )

                # Update individual subplot axis labels
                fig.update_xaxes(title_text="資料索引", row=1, col=1)
                fig.update_yaxes(title_text="數值", row=1, col=1)
                fig.update_xaxes(title_text="資料索引", row=1, col=2)
                fig.update_yaxes(title_text="數值", row=1, col=2)

                # Display the figure in Streamlit
                st.plotly_chart(fig)

                time.sleep(0.5)
                my_bar.progress((percent_complete + 1) / len(options), text=progress_text)
            time.sleep(1)
            my_bar.empty()

            st.success("所有模型已訓練完成，可至「試試手」頁面選擇指定圈數的模型進行預測分析！")
            

# 分頁 2: 試試手
elif page == "試試手":
    st.title("使用模型進行分析")

    models_dir = 'models'
    model_files = os.listdir(models_dir)

    # Use regular expression to extract numbers from filenames
    model_numbers = []
    for file in model_files:
        match = re.search(r'lr_model_(\d+)\.pkl', file)
        if match:
            model_numbers.append(int(match.group(1)))
    model_numbers.sort()
    option = st.selectbox(
        "預測模型種類(圈數)",
        tuple(model_numbers),
    )

    if len(model_numbers) > 0 and model_numbers:
        circle_no = int(option)
        with open(os.path.join(models_dir, f'lr_model_{circle_no}.pkl'), 'rb') as f:
            model = pickle.load(f)

        txt_uploaded_file = st.file_uploader("上傳設定檔案 (TXT)", type=["txt"])        
        uploaded_settings = {
            "P1": None, 
            "C1": None, 
            "C2": None, 
            "C3": None, 
            "C4": None, 
            "C5": None, 
            "E": None, 
            "F": None, 
            "G": None
        }
        
        if txt_uploaded_file is not None:
            content = txt_uploaded_file.getvalue().decode("utf-8")
            for line in content.splitlines():
                key, value = line.split()  # 分割每一行
                uploaded_settings[key] = float(value)
        
        # 2. 設定各個欄位 (PC1, PC2, PC3, Anode_1~5, Cathode_1~5)
        st.subheader("設定配方值(手動/上傳設定檔)")
        col1, col2, col3 = st.columns(3)

        with col1:
            P1_ = st.slider('配方P1 (93%-98%)', 93.0, 98.0, uploaded_settings['P1'], step=0.1)
            C3_ = st.slider('配方C3 (0%-10%)', 0.0, 10.0, uploaded_settings['C3'], step=0.1)
            E_ = st.slider('配方E (1%-2.5%)', 1.0, 2.5, uploaded_settings['E'], step=0.1)        
        
        with col2:
            C1_ = st.slider('配方C1 (0%-10%)', 0.0, 10.0, uploaded_settings['C1'], step=0.1)
            C4_ = st.slider('配方C4 (0%-10%)', 0.0, 10.0, uploaded_settings['C4'], step=0.1)
            F_ = st.slider('配方F (1%-2.5%)', 1.0, 2.5, uploaded_settings['F'], step=0.1)
        
        with col3:
            C2_ = st.slider('配方C2 (0%-10%)', 0.0, 10.0, uploaded_settings['C2'], step=0.1)
            C5_ = st.slider('配方C5 (0%-10%)', 0.0, 10.0, uploaded_settings['C5'], step=0.1)
            G_ = st.slider('配方G (1%-2.5%)', 1.0, 2.5, uploaded_settings['G'], step=0.1)

        check = np.array([P1_, C1_, C2_, C3_, C4_, C5_])
        valid_conditions = sum(check) == 100.0

        if not valid_conditions:
            st.error(f"前六項配方總和為:{sum(check)}，必須等於 100")
        
        if valid_conditions:
            st.info("所有欄位皆符合規範，請點擊下方按鈕開始分析。")
            if st.button("執行分析"):
                st.write(f"配方P1: {P1_}% / 配方C1: {C1_}% / 配方C2: {C2_}% / 配方C3: {C3_}% / 配方C3: {C4_}% / 配方C3: {C5_}% / 配方E: {E_}% / 配方F: {F_}% / 配方G: {G_}%")
                        
                # 構建輸入數據進行預測，以配方組合計算最接近的資料，並輸出其index
                df = pd.read_csv('input_V3.0.csv')
                df['battery_capacity'] = df['battery_capacity'].apply(ast.literal_eval)
                final_settings = {
                    "P1": P1_, 
                    "C1": C1_, 
                    "C2": C2_, 
                    "C3": C3_, 
                    "C4": C4_, 
                    "C5": C5_, 
                    "E": E_, 
                    "F": F_, 
                    "G": G_
                }
                mse = ((df[["P1", "C1", "C2", "C3", "C4", "C5", "E", "F", "G"]] - pd.Series(final_settings)) ** 2).mean(axis=1)
                closest_index = mse.idxmin()
                
                # 5. 顯示預測結果
                st.markdown(f"電池最終電量比例預測結果 (Final_Battery_Percent): <span style='color: green; font-weight: bold;'>{df['battery_capacity'].iloc[closest_index][-1] * 100:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.warning("請確保所有欄位的總和條件都符合規範。")
    else:
        st.error("未正確讀取已訓練之模型檔案，請先訓練模型。")
