# tf-sentiment-docker

## Docker Setup
0. Install [Docker](https://docs.docker.com/engine/installation/)
1. Run `git clone https://github.com/goodrahstar/tf-sentiment-docker`
2. Open docker terminal and navigate to `/path/to/tf-sentiment-docker`
3. Run `docker build -t sentiment-api .`
4. Run `docker run -p 8180:8180 sentiment-api`
5. Access http://0.0.0.0:8180/sentiment?message="i love it" from your browser [assuming you are on windows and docker-machine has that IP. Otherwise just use localhost]

## Native Setup
1. Anaconda distribution of python 2.7
2. `pip install -r requirements.txt`
3. Some dependencies from *nltk* (`nltk.download()` from python console and download averaged perceptron tagger)

### Run it
1. Place the script in any folder
2. Open command prompt and navigate to that folder
3. Type "python CLAAS.py"and hit enter
4. Go over to http://localhost:8180/apidocs/index.html in your browser (preferably Chrome) and start using.
