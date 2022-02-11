import cog
import json
import urllib
import notebook_v2 as notebook # ./notebook_v2.py

class Predictor(cog.Predictor):
    def setup(self):
      print("hello, setup")

    @cog.input("username", type=str, help="GitHub username")
    def predict(self, username):
      href = f"https://github.com/{username}.png"
      filename = f"{username}.png"
      urllib.request.urlretrieve(href, filename)
      feature_vector = notebook.file_to_feature_vector(filename)
      prediction = notebook.clf.predict([feature_vector])[0]

      output = json.dumps({
        "username": username, 
        "href": href, 
        "prediction": "default" if prediction == 1 else "custom"
      }, indent=2)

      return output
