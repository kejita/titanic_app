This is kejita's git.    
Thank you for your coming this space!

[about me]    
my job => data analyst    
to be => data engineer    
studying => python & ML & AI    

---

ICU転棟リスク予測アプリの使い方

- 実行コマンド: `streamlit run icu_app.py`
- 入力: 6時間ごとの時系列CSV（例: `患者データサンプル.csv`）
- 列マッピング: 画面で日本語列を選択（患者IDは任意、ラベル列はあれば評価可能）
- 教師ありモード: ラベル列（例: `ICU転棟_24h` が 0/1）を指定すると学習・評価（ROC/PR、感度/特異度、アラート比率）
- 教師なしモード: ラベルがない場合、IsolationForestによる異常度ベースのリスクを算出
- 出力: 予測付きCSVのダウンロード、時系列のリスク可視化

