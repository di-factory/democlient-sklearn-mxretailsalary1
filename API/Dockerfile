

FROM python:3.8-slim

WORKDIR /app

ADD API /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "/home/jagpascoe/democlient-sklearn/dif-s-mxretailsalary1/API/ml_profiling_best_API.py"]
