from flask import Flask, render_template, request, url_for
import os
from models import get_prediction

app = Flask(__name__)
# UPLOAD_FOLDERはstatic配下じゃないと、テンプレートから直接読めないのよ。分かった？
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'ファイルが選択されてないじゃない！'
        file = request.files['file']
        if file.filename == '':
            return 'ファイル名が空よ！'
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # モデルで予測を実行
            prediction = get_prediction(filepath)
            
            # 結果ページに画像URLと予測結果を渡すのよ
            image_url = url_for('static', filename='uploads/' + filename)
            return render_template('result.html', prediction=prediction, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
