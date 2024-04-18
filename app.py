import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots

from catboost import CatBoost,Pool
from catboost import CatBoostClassifier

import shap
import os
# 訓練したモデルを保存したfileを読み込むmodule
import pickle

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("機械学習を用いたタイタニック号乗客の生存確率分析")
    st.write("最終更新日: 2024/4/18")

    # サイドバーのmenu
    menu = ["分析概要", "分析目的", "予測結果", "生存要因", "結論"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定
    # 訓練済みのモデルファイル
    model_file = "./catmodel.pkl"
    # 読み込めているかを確認
    is_model_file = os.path.isfile(model_file)

    # 前処理前のtrain_data
    train_file = "./train.csv"
    # 前処理前のtest_data
    test_file = "./test.csv"
    # 前処理済みの説明変数のtraindata
    x_train_file = "./X_train.csv"
    # 前処理済みの説明変数のtestndata
    x_test_file = "./X_test.csv"
    # 前処理済みの説明変数のtraindata
    y_train_file = "./y_train.csv"
    # 読み込めているかを確認
    is_train_file = os.path.isfile(train_file)
    is_test_file = os.path.isfile(test_file)
    is_x_train_file = os.path.isfile(x_train_file)
    is_x_test_file = os.path.isfile(x_test_file)
    is_y_train_file = os.path.isfile(y_train_file)

    # モデルを再学習するかどうか
    # 再学習しないことを宣言
    no_update = True

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    print(is_train_file)
    print(is_test_file)
    print(is_x_train_file)
    print(is_x_test_file)
    print(is_y_train_file)
    print(no_update)
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    X_train = pd.read_csv(x_train_file)
    X_test = pd.read_csv(x_test_file)
    y_train = pd.read_csv(y_train_file)
    print("データを読み込みました")


    cat_model = pickle.load(open(model_file, 'rb'))
    print("モデルが読み込まれました")
    
    # 読み込んだモデルでsurvivedを予測
    pred = cat_model.predict(X_test)

    # 読み込んだモデルで生存確率を予測
    pred_prob = cat_model.predict(X_test, prediction_type="Probability")
    pred_prob = np.round(pred_prob, 3)

    # menuの中身
    # 分析の概要
    if chosen_menu == "分析概要":
        st.subheader("分析概要")
        st.write(" 1912年に沈没したタイタニック号の乗客データに基づいて、生存者と非生存者を予測する機械学習モデルを作成しました。")
        st.subheader("データセットの内容")
        st.write("訓練データには、891人の乗客情報 (生存フラグ、名前、年齢、性別、社会階級、乗船港など)が含まれます。")
        st.write("テストデータには、418人の乗客情報(生存フラグ以外)が含まれます。")
        st.write("訓練データを用いてモデルを構築して、テストデータの生存フラグ(Survived)を予測しました。")
        st.write(" ")
        st.write(" ")
        st.text("訓練データ")
        st.dataframe(train_df.head())
        st.write(" ")
        st.text("テストデータ")
        st.dataframe(test_df.head())


    # データセットの概要
    elif chosen_menu == "分析目的":
        st.subheader("分析目的")
        st.write("・乗客の属性情報 (年齢、性別、社会階級など) を基に、生存確率を予測すること")
        st.write("・生存確率に影響を与えた要因を分析すること")

    elif chosen_menu == "予測結果":
        st.subheader("予測結果")
        # 生存割合の可視化
        # y_testsから各カテゴリの出現頻度を計算
        survived_count = np.sum(pred)
        no_survived_count = len(pred) - survived_count

        # 整数型に直す
        survived_count = int(survived_count)
        no_survived_count = int(no_survived_count)
        values = [no_survived_count, survived_count]

        # 目的変数(customerstatus)の分布を確認
        type_ = ["No Survived=0", "Survived=1"]

        fig = make_subplots(rows=1, cols=1)

        # 円グラフの設定
        fig.add_trace(go.Pie(labels=type_, values=values, name="Survived"))

        # 円グラフにholeとhover機能をつける
        fig.update_traces(hole=0.4, hoverinfo="label+percent", textfont_size=16)


        # pieのレイアウトを設定
        fig.update_layout(
            # タイトル
            title_text = "<b>予測したテストデータのSurvivedの分布<b>",
            # pieの中央にsurvivedと表示させる
            annotations = [dict(text="Survived", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        # streamlitで表示
        st.plotly_chart(fig)

        # 生存確率の分布を可視化
        fig = px.histogram(pred_prob[:, 1])

        # 各図をsubplotsで表示
        fig = go.Figure(data=fig)
        fig.update_layout(height=600, 
                        showlegend=False, 
                        title="<b>生存確率の分布<b>", 
                        # タイトルの位置
                        xaxis_title="生存確率", 
                        yaxis_title="頻度"
                        )
        # streamlitで表示
        st.plotly_chart(fig)

        # 結果についての説明
        st.write("精度スコアから、作成したモデルは未知のデータに対して83%の精度で予測できることが分かりました。")
        st.write("ROC AUCスコアが0.81であり、再現率と適合率も比較的良好なモデルが作成できました。")
        st.write("確率分布から生存確率が0から1までの間に広く分布していることは、モデルがあるデータポイントの生存する可能性を様々に評価していることを示唆しています。")
        st.write("これは、モデルがデータに適切に適合し、高い汎化性能を持つ可能性があることを示唆しています。")
    elif chosen_menu == "生存要因":
        st.subheader("SHAP分析の結果")
        st.write("SHAP値とは、乗客の属性情報 (年齢、性別、社会階級など)がモデルの予測にどの程度貢献しているかを数字で表しています。")

        # shap分析の結果を表示
        st.write("SHAP値は、各特徴量が予測に与える影響度を表す指標です。図では、中央線が0を表しており、左側が死亡、右側が生存を表します。")
        st.write("例えば、Age(年齢): 年齢が高くなるほど、生存確率は低くなります。")
        st.write("また、Family(同行者数): 同行者数が多いほど、死亡確率は高くなります。")
        st.write("[注意]SHAP値は、定量的な属性情報の数字的な強さを表すことができますが、定性的な属性情報の数字的な強さを表すことはできません。")
        st.write("例えば、性別は定性的な属性情報であり、生存予測において重要な役割を果たしたことが確かです。しかし、女性が男性よりも助かりやすかったという因果関係は直接的には把握できません。")
        # shapの可視化
        pool = Pool(X_train, y_train)
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(pool)
        fig = shap.summary_plot(shap_values, X_train)
        st.pyplot(fig)

    elif chosen_menu == "結論":
        st.subheader("結論")
        st.write("・Sex(性別)やPclass(船室料金)が生存確率に大きな要因を与えました。")
        st.write("・ご高齢の方が優先的に助けられたと考えられます。")
        st.write("・同行者数が多いほど、救助は後回しにされたと考えられます。")


# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
