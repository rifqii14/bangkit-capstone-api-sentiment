# Repository API Machine Learning Model Sentiment Analysis Covid-19 Vaccines (Bangkit Capstone Project)

### General Description
This is API to access Machine Learning Model Sentiment Analysis Covid-19 Vaccines.
Created using FastAPI.


### How to Run API
1. You need to install these libraries using pip3
```
pip3 install tensorflow
pip3 install pydantic
pip3 install fastapi
pip3 install uvicorn
```

2. To start the server, please run (for example)
```
uvicorn main:app --host 127.0.0.1 --port 9000
```

3. Access API
- To Access API docs please open 
```
127.0.0.1:9000/docs
```
- To Access API predict sentiment please access using POST method and using JSON for request
```
127.0.0.1:9000/predict

Example Input (JSON) :
{
  "text_twitter": "Saya suka vaksin"
}

Example Output (JSON) :
{
    "result": "Positive"
}

```