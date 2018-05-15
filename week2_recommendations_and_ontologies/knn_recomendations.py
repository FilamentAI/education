from surprise import KNNBasic
from surprise import Dataset

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

algo = KNNBasic()
algo.fit(trainset)

uid = str(196)
iid = str(302)
algo.predict(uid, iid, r_ui=4, verbose=True)
