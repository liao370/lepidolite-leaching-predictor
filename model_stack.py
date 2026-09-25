"""Grouped out-of-fold stacking; all meta-features are from unseen input groups."""
import numpy as np,pandas as pd,hashlib
from sklearn.base import BaseEstimator,RegressorMixin,clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
_CACHE={}

class GroupSafeStacking(RegressorMixin,BaseEstimator):
    def __init__(self,alpha=1.,passthrough=False):self.alpha=alpha;self.passthrough=passthrough
    def _bases(self):
        return [LGBMRegressor(n_estimators=180,num_leaves=15,min_child_samples=8,learning_rate=.05,verbosity=-1,n_jobs=2,random_state=492),XGBRegressor(n_estimators=180,max_depth=3,learning_rate=.05,n_jobs=2,random_state=492,tree_method='hist'),ExtraTreesRegressor(n_estimators=150,max_features=.8,min_samples_leaf=2,n_jobs=2,random_state=492),Pipeline([('scale',StandardScaler()),('svr',SVR(C=60,epsilon=.5))])]
    def fit(self,X,y):
        X=np.asarray(X);y=np.asarray(y);self.n_features_in_=X.shape[1]
        g=pd.util.hash_pandas_object(pd.DataFrame(X),index=False).to_numpy()
        cv=GroupKFold(n_splits=5,shuffle=True,random_state=492)
        key=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
        if key in _CACHE:
            meta,self.estimators_=_CACHE[key]
            meta=meta.copy()
        else:
            meta=np.zeros((len(X),4));self.estimators_=[]
            for j,base in enumerate(self._bases()):
                for tr,va in cv.split(X,y,g):
                    b=clone(base).fit(X[tr],y[tr]);meta[va,j]=b.predict(X[va])
                self.estimators_.append(clone(base).fit(X,y))
            _CACHE[key]=(meta.copy(),self.estimators_)
        if self.passthrough:meta=np.column_stack([meta,X])
        self.final_estimator_=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=self.alpha))]).fit(meta,y)
        return self
    def predict(self,X):
        X=np.asarray(X);meta=np.column_stack([b.predict(X) for b in self.estimators_])
        if self.passthrough:meta=np.column_stack([meta,X])
        return self.final_estimator_.predict(meta)
