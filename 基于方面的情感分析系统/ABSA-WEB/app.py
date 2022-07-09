# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

from flask import Flask, request, render_template, jsonify
import analysis

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('ABSA.html')


@app.route('/get_analysis.do')
def get_analysis():
    text = request.args.get('text')
    asp_l = int(request.args.get('asp_l'))
    asp_r = int(request.args.get('asp_r'))
    text_words = text.lower().strip().split()
    text_len = len(text_words)

    if asp_r < asp_l:
        return jsonify({'check': 'r less than l'})

    if text_len > 85:
        return jsonify({'check': 'too long'})

    if asp_l > text_len or asp_r > text_len:
        return jsonify({'check': 'aspect position overflow'})

    polarity = analysis.predict(text, asp_l-1, asp_r-1)
    weights = analysis.get_weights(text_words, asp_l-1, asp_r-1)
    return jsonify({'check': 'ok', 'polarity': polarity, 'xtick': text_words, 'weights': weights})


if __name__ == '__main__':
    app.run()
