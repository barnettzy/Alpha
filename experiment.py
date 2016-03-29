import loaddata
import PCC_similarity as sim
import model
import PredictMissingValue as pmv

dataset = loaddata.load_datasets('movielens100k.data')
datamat = model.MatrixFromData(dataset.data)
pd = pmv.Prediction(datamat)


file_ = open('data.dat', 'w')

print datamat
for i in range(datamat.user_count()):
    user = datamat.preference_values_from_user(i+1)
    for c in user[0]:
        file_.write(str(c) + ' ')
    file_.write('\n')
file_.flush()
file_.close()
print 'done'
