import json
import urllib

import joblib
from cog import BasePredictor, Input
from preprocess import file_to_feature_vector

class Predictor(BasePredictor):
    def setup(self):
      print("loading pickled classifier")
      self.clf = joblib.load("model.pkl")

    def predict(self, 
      username: str = Input(description="GitHub username")
    ) -> str:
      # load the image
      href = f"https://github.com/{username}.png"
      filename = f"{username}.png"
      urllib.request.urlretrieve(href, filename)
      
      # create a feature vector from the image
      feature_vector = file_to_feature_vector(filename)

      # classify that mofo
      prediction = self.clf.predict([feature_vector])[0]

      output = json.dumps({
        "username": username, 
        "href": href, 
        "prediction": "default" if prediction == 1 else "custom"
      }, indent=2)

      return output
