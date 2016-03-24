import loaddata
import PCC_similarity as sim
import model
import PredictMissingValue as pmv

dataset = loaddata.load_datasets('movielens100k.data')
datamat = model.MatrixFromData(dataset.data)
pd = pmv.Prediction(datamat)

print datamat
