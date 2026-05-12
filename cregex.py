from utils import *
from fregex import FREGEX
from bert import BERT
from mysetfit import SETFIT
from zeroshot import ZSL

class CREGEX(object):
    def __init__(self, 
                FILENAME, MODEL_NAMES, N_CLASSES, LANG, THR_CONF_CLF_opt = True,
                PROBS_THR=np.arange(0.55, 1, 0.05),
                NGRAM_MIN=NGRAM_MIN, pnumbers=pnumbers, 
                gap_cmb=gap_cmb, whitespaces=whitespaces, lexicon=lexicon, HYPERPARAMS=HYPERPARAMS, SEED=SEED):
        self.__metaclass__ = 'CREGEX'
        self.FILENAME = FILENAME
        clfs, _ = MODEL_NAMES.split('*')
        clfs = clfs.split('.')
        self.MODEL_NAMES = [clf for clf in clfs]
        self.N_CLASSES = N_CLASSES
        self.NGRAM_MIN = NGRAM_MIN
        self.pnumbers=pnumbers
        self.gap_cmb = gap_cmb
        self.whitespaces = whitespaces
        self.lexicon = lexicon
        self.SEED = SEED
        self.HYPERPARAMS = HYPERPARAMS
        self.regexes = {}
        self.labeled_regexes = {}
        self.kw = []
        self.distribution = defaultdict(list)
        self.y = defaultdict(list)
        #print('self.y', self.y)
        self.rndm = np.random.RandomState(self.SEED)
        self.models = {}
        self.THR_CONF_CLF_opt = THR_CONF_CLF_opt
        self.THR_CONF_CLFS = {}
        self.PROBS_THR = PROBS_THR
        self.LANG = LANG

    def fit(self, X,y, X_val, y_val):
        print('CREGEX...fit')
        fregex = FREGEX(X, y, self.FILENAME, self.LANG)
        fregex.fit()
        self.regexes.update( fregex.transform() )        
        self.kw = copy.deepcopy(self.lexicon[self.FILENAME])
        self.pattern2token = copy.deepcopy( fregex.pattern2token )
        self.token2pattern = copy.deepcopy( fregex.token2pattern )
        self.tokens2pos = copy.deepcopy( fregex.tokens2pos )
        self.stopwords = copy.deepcopy(fregex.stopwords)
        self.regexes, self.regex2class = get_classes_regexes(self.regexes, y, self.tokens2pos)
        
        self.kw = [k[:3] for k in self.kw]
        self.exclusiones = {}
        self.window = 2
        self.limit = rf"(?:\s+\S+){{0,{self.window}}}\s+"
        split_pat = r'(?:%s|%s)' % (re.escape(self.gap_cmb), re.escape(self.whitespaces))
        
        for rgx_a in self.regexes:
            label_a = self.regex2class[rgx_a][0]
            tkn_vecinos = set()
            
            # Pre-split de A para no repetirlo en el bucle interno
            tokens_a = [t for t in re.split(split_pat, rgx_a) if t.strip()]
            len_a = len(tokens_a)
            
            for rgx_b in self.regexes:
                if rgx_a == rgx_b: continue
                
                label_b = self.regex2class[rgx_b][0]
                
                # Si las etiquetas son distintas y el string de A está contenido en B
                if label_a != label_b and rgx_a in rgx_b:
                    tokens_b = [t for t in re.split(split_pat, rgx_b) if t.strip()]
                    len_b = len(tokens_b)
                    
                    # Buscamos la posición exacta de la secuencia de tokens_a en tokens_b
                    for i in range(len_b - len_a + 1):
                        if tokens_b[i : i + len_a] == tokens_a:
                            # Extraemos ventana de 3 tokens antes y 3 después
                            tkn_vecinos.update(tokens_b[max(0, i - self.window) : i])
                            tkn_vecinos.update(tokens_b[i + len_a : i + len_a + self.window])
                            break # Encontrado el conflicto, saltamos a la siguiente rgx_b
        
            if tkn_vecinos:
                tkn_vecinos = list(tkn_vecinos)
                tkn_vecinos = [t for t in tkn_vecinos if self.pnumbers not in t]
                self.exclusiones[rgx_a] = tkn_vecinos
                
        
        # --- Fase de Validación para el Score ---
        for rgx, data in self.regexes.items():
            _, n_aux, _, _, _ = data
            label_obj, _ = self.regex2class[rgx]
            
            tp, fp, fn = 0, 0, 0
            for i, text in enumerate(X):
                if findall(rgx, [], n_aux, text):
                    bloqueado = False
                    if rgx in self.exclusiones and len(n_aux)==0:
                        for tkn in self.exclusiones[rgx]:
                            #n_excl = [] #n_aux if any(str(n) in tkn for n in n_aux) else []
                            patron_compuesto = rf"{tkn}{self.limit}{rgx}|{rgx}{self.limit}{tkn}"
                            if findall(patron_compuesto, [], n_aux, text):
                                bloqueado = True
                                break
                    
                    if not bloqueado:
                        if y[i] == label_obj:
                            tp += 1
                        else:
                            fp+=1
            
            # Guardamos [label, precision_neta]
            score = tp / (tp+fp) if (tp+fp) > 0 else 0
            if score > THR_CONF:
                tokens_rgx = [fregex.pattern2token.get(t, t) for t in re.split(split_pat, rgx) if t.strip()]
                if any(t[:3] in self.kw for t in tokens_rgx):
                    self.labeled_regexes[rgx] = [label_obj, score]
        #######################################################################

        keys = copy.deepcopy( list( self.regexes.keys() ) )
        for key in keys:
            if key not in self.labeled_regexes:
                self.regexes.pop(key)
                self.regex2class.pop(key)

        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME+'...fit')
            tokens = None
            opt = None
            regexes_aux = None
            model = None
            X_val_aux = None
            y_val_aux = None
            X_l_aux = None
            y_l_aux = None
            if 'random' not in MODEL_NAME: #clf                
                seed_everything()
                if 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME and 'zsl' not in MODEL_NAME:
                    _, NGRAM_SIZE = MODEL_NAME.split('-')
                    NGRAM_SIZE = int(NGRAM_SIZE.replace('n',''))
                    tokens = n_grams(X, NGRAM_SIZE)
                    if 'F' in MODEL_NAME:
                        tokens = [t for t in tokens if any(t[:3] == k[:3] for k in self.kw)]
                    opt = False
                    regexes_aux = {}
                    y_l_aux = copy.deepcopy(y)                    
                    X_l_aux = copy.deepcopy( get_matrix(tokens, X, regexes_aux, opt) )
                    X_val_aux = copy.deepcopy( get_matrix(tokens, X_val, regexes_aux, opt) )
                    y_val_aux = copy.deepcopy(y_val)
                    X_train_val = copy.deepcopy( np.vstack((X_l_aux, X_val_aux)) )            
                    ps = PredefinedSplit( np.array( [0]*len(y)+[-1]*len(y_val) ) )
                    y_train_val = copy.deepcopy( np.hstack((y, y_val_aux)) )
                    HYPERPARAMS = best_model(MODEL_NAME, ps, X_train_val, y_train_val)
                    model = select_trad_model(MODEL_NAME, HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME] = copy.deepcopy(HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME+'-cregex'] = copy.deepcopy(HYPERPARAMS)
                else:
                    X_l_aux = copy.deepcopy(X)
                    y_l_aux = copy.deepcopy(y)
                    X_val_aux = copy.deepcopy(X_val)
                    y_val_aux = copy.deepcopy(y_val)
                    if 'bert' in MODEL_NAME:
                        model = BERT(**self.HYPERPARAMS['bert'])
                    elif 'setfit' in MODEL_NAME:
                        model = SETFIT(**self.HYPERPARAMS['setfit'])
                    elif 'zsl' in MODEL_NAME:
                        model = ZSL(**self.HYPERPARAMS['zsl'])
                model.fit(X_l_aux, y_l_aux)
                pred_val = model.predict_proba(X_val_aux)
            else:
                X_val_aux = copy.deepcopy(X_val)
                y_val_aux = copy.deepcopy(y_val)
                seed_everything()
                pred_val = []
                for _ in range(len(X_val_aux)):
                    pred = self.rndm.randint(0, self.N_CLASSES) 
                    pond = 1/(self.N_CLASSES+1)
                    preds = np.ones(self.N_CLASSES)*pond
                    preds[pred] = 1-pond*(self.N_CLASSES-1)
                    pred_val.append(preds)
                pred_val = np.array(pred_val)

            if self.THR_CONF_CLF_opt:
                precision, recall, thresholds, weights = prec_rec_curves(y_val_aux, pred_val, self.PROBS_THR, self.N_CLASSES)
                if self.N_CLASSES<3:
                    precision[np.isnan(precision)] = 0
                    recall[np.isnan(recall)] = 0
                    fscore = (2 * precision * recall) / (precision + recall)
                    fscore[np.isnan(fscore)] = 0
                    idx = np.argmax(fscore)
                    self.THR_CONF_CLFS[MODEL_NAME] = thresholds[idx]
                else:
                    aux_thr = []
                    for c in range(self.N_CLASSES):
                        precision[c][np.isnan(precision[c])] = 0
                        recall[c][np.isnan(recall[c])] = 0
                        fscore = (2 * precision[c] * recall[c]) / (precision[c] + recall[c])
                        fscore[np.isnan(fscore)] = 0
                        idx = np.argmax(fscore)
                        aux_thr.append(thresholds[c][idx])
                    self.THR_CONF_CLFS[MODEL_NAME] = aux_thr

                self.distribution['thresholds-'+MODEL_NAME] = [precision, recall, fscore, thresholds, weights] 
                #self.distribution['thresholds-'+MODEL_NAME].append( [precision, recall, fscore, thresholds, weights] )
                #self.THR_CONF_CLF = thresholds[idx]
                #print('THR_CONF_CLF', self.THR_CONF_CLF)
                #fig = pylab.figure(1)
                #pylab.plot(recall, precision)
                #pylab.show()
                #print('idx', precision[idx], recall[idx], fscore[idx])

            self.models[MODEL_NAME] = [tokens, opt, regexes_aux, model, X_l_aux, y_l_aux]            
            
    def predict(self, X): #always predict_proba
        self.y = defaultdict(list)
        thresholds = {k: v for k, v in self.distribution.items() if 'thresholds-' in k}
        self.distribution.clear()
        self.distribution.update(thresholds)
        del thresholds
    
        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME+'...predict')
            tokens, opt, regexes_aux, model, X_train, y_train = self.models[MODEL_NAME]
            if 'random' not in MODEL_NAME: #-clf
                if 'bert' not in MODEL_NAME and 'setfit' not in MODEL_NAME and 'zsl' not in MODEL_NAME:
                    X_test_aux = copy.deepcopy( get_matrix(tokens, X, regexes_aux, opt) )
                else:
                    X_test_aux = copy.deepcopy(X)
                predictions = model.predict_proba( X_test_aux )    
                predictions = [list(p) for p in predictions]     
                #print('predictions', predictions) 
            elif 'random' in MODEL_NAME:
                seed_everything()
                predictions = []
                for _ in range(len(X)):
                    pred = self.rndm.randint(0, self.N_CLASSES) 
                    pond = 1/(self.N_CLASSES+1)
                    #preds = np.ones(self.N_CLASSES)*pond
                    preds = list(np.ones(self.N_CLASSES)*pond)
                    preds[pred] = 1-pond*(self.N_CLASSES-1)
                    predictions.append(preds)
                #predictions = np.array(predictions)

            self.y[MODEL_NAME] = copy.deepcopy(predictions) #list(predictions)

            #print('self.y', self.y)

        print('CREGEX...predict')
        #print('y', self.y)
        i = -1     
        for text in X:
            i+=1
            classe = None
            for MODEL_NAME in self.MODEL_NAMES:
                #print('MODEL_NAME', self.y[MODEL_NAME][i])
                if self.N_CLASSES<3:
                    pos_pred = self.y[MODEL_NAME][i][1]
                    THR_PROB = self.THR_CONF_CLFS[MODEL_NAME]
                else:
                    c = np.argmax(self.y[MODEL_NAME][i])
                    pos_pred = self.y[MODEL_NAME][i][c]
                    THR_PROB = self.THR_CONF_CLFS[MODEL_NAME][c]

                if pos_pred>=THR_PROB:
                    classe = copy.deepcopy( self.y[MODEL_NAME][i] )
                    #print(type(classe), 'A')
                    self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('clf-A', None) )
                    
                else:

                    flag = False
                    labels = []
                    confs = []
                    regexs = []
                    max_conf = []
                    regexs_labels = []
                    rlc = []

                    for regex in self.labeled_regexes:
                        label, conf = self.labeled_regexes[regex]
                        # Extraemos los datos de la regex base
                        _, numbers_aux, _, _, _ = self.regexes[regex]
                        
                        # 1. Primer findall: Match de la regex corta
                        f = findall(regex, [], numbers_aux, text)
                        
                        if f:
                            bloqueado = False
                            if regex in self.exclusiones and len(numbers_aux)==0:
                                for tkn in self.exclusiones[regex]:
                                    # 2. Segundo findall: 
                                    # Si el pnumber (numbers_aux) está en el texto del token, se lo pasamos
                                    # Si no, mandamos lista vacía.
                                    #n_aux_excl = [] #numbers_aux if any(str(n) in tkn for n in numbers_aux) else []
                                    patron_compuesto = rf"{tkn}{self.limit}{regex}|{regex}{self.limit}{tkn}"
                                    if findall(patron_compuesto, [], numbers_aux, text):
                                        bloqueado = True
                                        break
                            
                            if not bloqueado:
                                flag = True
                                labels.append(label)
                                confs.append(conf)
                                regexs.append(regex)
                    
                    if flag:
                        regexs = np.array(regexs)
                        labels = np.array(labels)
                        confs = np.array(confs)
                        
                        rlc = list(zip(regexs, labels, confs))                
                        eps = 1e+4
                        rlc = sorted( rlc, 
                                key=lambda x:x[2]+len( re.split(r'(?:%s|%s)' %(re.escape(self.gap_cmb), re.escape(self.whitespaces)), x[0]))/eps, 
                                reverse=True)
                        
                        classe = copy.deepcopy( rlc[0][1] )
                        max_conf = rlc[0][2]
                        pos_aux = copy.deepcopy(classe)
                        #pond = (1-max_conf)/( predictions.shape[1]-1)
                        pond = (1-max_conf)/( self.N_CLASSES-1)
                        #classe =  np.ones(predictions.shape[1])*pond
                        #classe =  np.ones(self.N_CLASSES)*pond
                        classe =  list(np.ones(self.N_CLASSES)*pond)
                        classe[ pos_aux ] = max_conf

                        #print(type(classe), 'R')

                        r_aux, l_aux, c_aux = zip(*rlc)

                        self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('rex', [list(r_aux), list(l_aux), list(c_aux)]) )

                    else:
                        classe = copy.deepcopy( self.y[MODEL_NAME][i] )
                        #print(type(classe), 'B')
                        self.distribution['predict-'+MODEL_NAME+'-cregex'].append( ('clf-B', None) )

                #print('self.y', self.y[MODEL_NAME+'-cregex'])
                self.y[MODEL_NAME+'-cregex'].append( classe )

        #esto es para results (clf base y cregex, main.py)
        for key in self.y:
            self.y[key] = np.array(self.y[key])
        #esto es para las lc (1 prediccion, curves.py)
        key = [k_ for k_ in self.y if 'cregex' in k_][0]
        return self.y[key]

    def predict_proba(self, X): #always predict_proba
        self.y = defaultdict(list)        
        thresholds = {k: v for k, v in self.distribution.items() if 'thresholds-' in k}
        self.distribution.clear()
        self.distribution.update(thresholds)
        del thresholds
        #self.distribution = defaultdict(list)
        #y = copy.deepcopy( self.predict(X) )
        #for key in y:
        #    y[key] = np.array(y[key])
        #
        #return y
        return copy.deepcopy( self.predict(X) )
