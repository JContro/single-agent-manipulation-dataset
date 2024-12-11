import pandas as pd
import json

with open("analytics_data.json", "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data)