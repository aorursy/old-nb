#!/usr/bin/env python
# coding: utf-8



import fastai
from fastai.vision import *




work_dir = Path('/kaggle/working/')
path = Path('../input')




train = 'train_images/train_images'
test =  path/'leaderboard_test_data/leaderboard_test_data'
holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'
sample_sub = path/'SampleSubmission.csv'
labels = path/'traininglabels.csv'




df = pd.read_csv(labels)
df_sample = pd.read_csv(sample_sub)




df.head()




df.describe()




df[df['score']<0.75]




(df.has_oilpalm==1).sum()




test_names = [f for f in test.iterdir()]
holdout_names = [f for f in holdout.iterdir()]




src = (ImageItemList.from_df(df, path, folder=train)
      .random_split_by_pct(0.2, seed=2019)
      .label_from_df('has_oilpalm')
      .add_test(test_names+holdout_names))




data =  (src.transform(get_transforms(), size=128)
         .databunch(bs=64)
         .normalize(imagenet_stats))




data.show_batch(3, figsize=(10,7))




#This was working perfectly some minutes ago!
from sklearn.metrics import roc_auc_score
def auc_score(preds,targets):
    return torch.tensor(roc_auc_score(targets,preds[:,1]))




learn = create_cnn(data, models.resnet18, 
                   metrics=[accuracy], #<---add aoc metric?
                   model_dir='/kaggle/working/models')




learn.lr_find(); learn.recorder.plot()




lr = 1e-2




learn.fit_one_cycle(6, lr)




# Then we unfreeze and train the whole model, with lower lr.




learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-3))




learn.save('128')




p,t = learn.get_preds()
auc_score(p,t)




interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()




interp.plot_top_losses(9, figsize=(15,11))




p,t = learn.get_preds(ds_type=DatasetType.Test)




p = to_np(p); p.shape




ids = np.array([f.name for f in (test_names+holdout_names)]);ids.shape




#We only recover the probs of having palmoil (column 1)
sub = pd.DataFrame(np.stack([ids, p[:,1]], axis=1), columns=df_sample.columns)




sub.to_csv(work_dir/'sub.csv', index=False)






