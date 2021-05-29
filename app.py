import uvicorn
from fastapi import FastAPI
from Model import SentimentModel, SentimentText

app = FastAPI()
model = SentimentModel()

@app.post('/predict')
def predict_sentiment(sentiment: SentimentText):
    data = sentiment.dict()
    result = model.predict_sentiment(
        data['text_twitter']
    )
    return {
        'result': result
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)