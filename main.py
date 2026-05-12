#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ast
import sys
import os

from mysnorkel import Snorkel
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()


#setfit
#1.1.13, 0.7.0 (hast/actune)

#Dale permisos: chmod +x main_curves.sh
#Ejecútalo: ./main_curves.sh

FILENAME = sys.argv[1]
min_idx = int( sys.argv[2] )
max_idx = int( sys.argv[3] )
MODELS = [sys.argv[4]]
CURVES = [sys.argv[5]]
RESULTS = ast.literal_eval( sys.argv[6] )


'''
FILENAME = 'OBESIDAD'
#FILENAME = 'FUMADOR'
#FILENAME = 'OBESIDAD_TIPOS'
#FILENAME = 'CODIESP'
#FILENAME = 'MIMIC'

min_idx = 1
max_idx = 1

MODELS = [
        'svmF-n1-cregex',
        'svm-n1-cregex',
        'rfF-n1-cregex',
        'rf-n1-cregex',
        'snorkel',
        'setfit-cregex',
        'bert-cregex',
        'zsl-cregex'
         ]

CURVES = ['PL', 'AL']

RESULTS = True
'''

if RESULTS:
    #MODELS = [m for m in MODELS if 'regex' in m]
    CURVES = ['RESULTS']

import warnings
warnings.filterwarnings("ignore")
import logging
logging.captureWarnings(True)
logging.disable(sys.maxsize)
from cregex import *
from utils import *
from curves import *
seed_everything()

N_CLASSES = {'FUMADOR':2, 'OBESIDAD':2, 'OBESIDAD_TIPOS':3, 'CODIESP':3, 'CARES':2, 'MIMIC':2}[FILENAME]
LANG = {'FUMADOR':'spanish', 'OBESIDAD':'spanish', 'OBESIDAD_TIPOS':'spanish', 'CODIESP':'spanish','MIMIC':'english'}
HYPERPARAMS['bert']['n_classes'] = N_CLASSES
HYPERPARAMS['setfit']['n_classes'] = N_CLASSES
HYPERPARAMS['zsl']['n_classes'] = N_CLASSES
HYPERPARAMS['zsl']['labels'] = labels[FILENAME]

create_paths(FILENAME)

with open( os.path.join( os.getcwd(), 'snippets_procesados_'+FILENAME),  'rb') as a:
    data = pickle.load(a)
    data = sorted(data, key = lambda x:x[0], reverse = False)
    DATA = np.array( [snippet for snippet, classe in data] )#[:150]
    CLASSES = np.array( [classe for snippet, classe in data])#[:150]

print(FILENAME)
RUNS = 1
FOLDS = 5
folds = KFold(n_splits = FOLDS, shuffle = False)
idxs = np.arange(0, len(DATA))

for r in range(RUNS):
    idxs = shuffle(idxs, random_state = SEED)
    CLASSES = CLASSES[idxs]
    DATA = DATA[idxs]
    k = -1
    for train_index, test_index in folds.split(idxs):

        X_train = copy.deepcopy( DATA[train_index] )
        y_train = copy.deepcopy( CLASSES[train_index] )
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        X_test = copy.deepcopy( DATA[test_index] )
        y_test = copy.deepcopy( CLASSES[test_index] )
        
        k+=1
        print('fold:', k+1)
        if (k+1) not in list(range(min_idx, max_idx+1)):
            continue

        for CURVE in CURVES:

            for MODEL in MODELS:

                #if 'PL' in CURVE and 'regex' in MODEL:
                #if 'PL' in CURVE or 'AL' in CURVE and 'snorkel' in MODEL:
                #    continue

                print('CURVE:', CURVE)
                print('MODEL:', MODEL)

                if 'snorkel' in MODEL:
                    curve = None
                else:
                    curve = Curves(
                        X_train, y_train,
                        X_val, y_val,
                        X_test,
                        N_CLASSES, CURVE, MODEL,
                        BATCH, FILENAME, LANG[FILENAME]
                    )

                if not RESULTS: #curve
                    curve.learningCurve()
                    results = [
                               curve.results['scores'],
                               curve.results['x'],
                               curve.results['y'],
                               curve.results['y_u_dst'],
                               curve.results['y_clf'],
                               curve.results['dst_cregex'],
                               curve.HYPERPARAMS,        
                               curve.N_FEATURES,        
                               y_test
                    ]
                    with open( os.path.join( os.getcwd(), 'out', 'RESULTSLC', CURVE, FILENAME, FILENAME+'_'+MODEL+'_'+CURVE+'_k'+str(k+1)+'.pkl' ), 'wb' ) as a:
                        pickle.dump(results, a, protocol=2)

                else: #results
                    pred = None
                    model = None
                    if 'snorkel' not in MODEL:
                        pred, _, __, model = curve.model_selection(X_train, y_train, X_test, X_u=[], results=False, return_model=True)
                    else:                        
                        model = Snorkel(lexicon[FILENAME], N_CLASSES)
                        model.fit(X_train, y_train, X_val, y_val)
                        pred = model.predict_proba(X_test)
                    if 'cregex' in MODEL:
                        results_y = copy.deepcopy(model.y)
                        dst_y = copy.deepcopy(model.distribution)
                        for key in results_y:
                            pred = results_y[key]
                            with open( os.path.join( os.getcwd(), 'out', 'RESULTS', FILENAME, FILENAME+'_'+key+'_'+'RESULTS'+'_k'+str(k+1)+'.pkl' ), 'wb' ) as a:
                                #pickle.dump([pred, times, dst_y, y_train, y_test,
                                pickle.dump([pred, dst_y, y_train, y_test,
                                             X_train, X_test], a, protocol=2)
                            print(key, 100*accuracy_score(y_test, pred.argmax(axis=1)))
                

                    else:

                        with open( os.path.join( os.getcwd(), 'out', 'RESULTS', FILENAME, FILENAME+'_'+MODEL+'_'+'RESULTS'+'_k'+str(k+1)+'.pkl' ), 'wb' ) as a:
                            pickle.dump([pred, None, y_train, y_test,
                                         X_train, X_test], a, protocol=2)
                    
                        print(MODEL, 100*accuracy_score(y_test, pred.argmax(axis=1)))

