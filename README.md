# Sentiment Classifier of Ukrainian Telegram Texts

*This project explores possible solutions for sentiment analysis of Ukrainian-language messages gathered from Telegram. Due to the lack of publicly available datasets, I created my own dataset, preprocessed the data, and compared the performance of various machine learning and deep learning models for classifying messages as positive or negative.* 

---

## Tools and Libraries



## Data Sources

The dataset was created using the official Telegram API to collect messages from the following news channels with open discussion:

| Channels usernames |
| ------------------ |
| @truexanewsua      |
| @UaOnlii           |
| @okoo_ukr          |
| @uniannet          |
| @ssternenko        |
| @DeepStateUA       |
| @bozhejakekonchene |
| @operativnoZSU     |
| @kyivoperat        |
| @karas_evgen       |


- **Timeframe**: January 2024
- **Source**: Top news channels with open discussions (according to Detector Media).
- **Total Messages**: 50,000
- **Annotated Messages**: 6,000 (positive and negative sentiment).
- **Training/Testing Split**: 80% training, 20% testing.

## Sentiment Classes

- **Positive**: Supportive, constructive criticism, or positive emotion without aggression/profanity.
- **Negative**: Criticism, aggression, profanity, or threats.

## Results

| Model         | Accuracy | Precision (Pos) | Precision (Neg) | Recall (Pos) | Recall (Neg) | F1-Score | AUC    |
| ------------- | -------- | --------------- | --------------- | ------------ | ------------ | -------- | ------ |
| SVM           | 86.16%   | 87%             | 85%             | 85%          | 88%          | 86%      | 94.27% |
| AdaBoost      | 80.84%   | 79%             | 83%             | 84%          | 78%          | 81%      | 80.85% |
| Random Forest | 83.54%   | 79%             | 90%             | 91%          | 76%          | 83%      | 89.68% |
| Naive Bayes   | 84.98%   | 88%             | 82%             | 81%          | 89%          | 85%      | 93.71% |
| LSTM          | 85.23%   | 84%             | 86%             | 87%          | 84%          | 85%      | 93.60% |

## Key Findings

- **SVM** achieved the highest accuracy (86.16%) and AUC (94.27%).
- Deep learning models like **LSTM** performed well but required more data for further improvement.
- **TF-IDF** and **GloVe** embeddings both proved effective for text representation.


