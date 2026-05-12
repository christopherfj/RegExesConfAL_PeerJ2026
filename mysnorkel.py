import numpy as np
import pandas as pd
from collections import Counter
# Volvemos a los imports que te funcionaron al principio
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel 
SEED = 42

class Snorkel:
    def __init__(self, keywords, cardinality, SEED=SEED):
        self.keywords = keywords
        self.cardinality = cardinality
        self.prefixes = list( set( [k[:3].lower() for k in keywords] ) )
        self.kw_to_class = {} 
        self.label_model = LabelModel(cardinality=cardinality, verbose=False)
        self.SEED = SEED

    def _get_best_class_for_kw(self, prefix, X, y):
        # Procesamos sobre los arrays
        occurrences = [label for text, label in zip(X, y) if prefix in str(text).lower()]
        return Counter(occurrences).most_common(1)[0][0] if occurrences else -1

    def fit(self, X_train, y_train, X_val=None, y_val=None):
            lfs = []
            default = Counter(y_train).most_common(1)[0][0]
            
            for pref in self.prefixes:
                assigned_class = self._get_best_class_for_kw(pref, X_train, y_train)
                if assigned_class != -1:
                    self.kw_to_class[pref] = assigned_class
                    
                    def make_lf(p, c, d):
                        @labeling_function(name=f"lf_{p}")
                        def _lf(x):
                            return c if p in str(x.text).lower() else d
                        return _lf
                    
                    lfs.append(make_lf(pref, assigned_class, default))

            self.applier = PandasLFApplier(lfs=lfs)
            L_train = self.applier.apply(pd.DataFrame(X_train, columns=["text"]))
            
            # AGREGAMOS lr=0.001 PARA EVITAR EL NaN
            if X_val is not None and y_val is not None:
                self.label_model.fit(L_train=L_train, Y_dev=y_val, n_epochs=500, lr=0.001, log_freq=100, seed=self.SEED)
            else:
                self.label_model.fit(L_train=L_train, n_epochs=500, lr=0.001, log_freq=100, seed=self.SEED)
                
            return self

    def predict_proba(self, X_test):
        L_test = self.applier.apply(pd.DataFrame(X_test, columns=["text"]))
        return self.label_model.predict_proba(L_test)

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=1)
