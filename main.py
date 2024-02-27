from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle




with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(data: InputText):
    try:
        # Vectorize the input text
        text_vector = vectorizer.transform([data.text])
        
        # Make a prediction
        prediction = model.predict(text_vector)[0]
        print(prediction)

        # Map the prediction to a sentiment label
        sentiment = "positive" if prediction == 1 else "negative"

        return {"sentiment": prediction}

    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))
    

    
    
class InputBatch(BaseModel):
    comments: list

class SentimentResponse(BaseModel):
    sentiments: list
    positive_count: int
    negative_count: int

def get_sentiment(comment):
    # Vectorize the input text
    text_vector = vectorizer.transform([comment])
    
    # Make a prediction
    prediction = model.predict(text_vector)[0]

    # Map the prediction to a sentiment label

    return prediction

@app.post("/predict_batch", response_model=SentimentResponse)
async def predict_batch_sentiment(data: InputBatch):
    try:
        sentiments = [get_sentiment(comment) for comment in data.comments]
        positive_count = sentiments.count("pos")
        negative_count = sentiments.count("neg")

        return {"sentiments": sentiments, "positive_count": positive_count, "negative_count": negative_count}

    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))