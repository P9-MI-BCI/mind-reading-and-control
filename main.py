import matlab.engine
import pandas as pd

eng = matlab.engine.start_matlab()

content = eng.load("dataset/Cue_Set1.mat", nargout=1)

df = pd.DataFrame(content['data_device1'])

print(df)