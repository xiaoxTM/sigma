import pickle
import gzip

########################################################################
def shuffle(database):
    """
    shuffle the given database
    """
    (samples, labels) = database

    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    return (samples[idx, :, :, :], labels[idx])


############################################################
def split(database, ratio):
    """
     split database into train, validate, and test subset according to
     ratio, which should have the form of [train_ratio, validate_ratio, test_ratio]
     and train_ratio + validate_ratio + test_ratio <= 1.0
     --> for sum of ratio less than 1, it means that only need sub-database and then
     --> split the sub-database into three sub-subset
    """
    assert len(ratio) == 3 and sum(ratio) <= 1
    nsamples = len(database[1])
    indice = np.int32(np.array(ratio) * nsamples)
    indice[1] = indice[0] + indice[1]
    indice[2] = indice[1] + indice[2]
    return ([database[0][:indice[0], :, :, :], database[1][:indice[0]]], \
            [database[0][indice[0]:indice[1],:,:,:], database[1][indice[0]:indice[1]]], \
            [database[0][indice[1]:indice[2],:,:,:], database[1][indice[1]:indice[2]]])

################################################
def pickle_database(name, db, **kwargs):
    """
    size as the same as function load
    """
    if name.endswith('gzip'):
        dummy = gzip.open(name,'wb')
    else:
        dummy = open(name, 'wb')
    pickle.dump(db, dummy, pickle.HIGHEST_PROTOCOL, **kwargs)
    dummy.close()


####################################
def load_pickle(name, **kwargs):
    """
    load from .pkl.gz
    """
    if name.endswith('gzip'):
        pkl = gzip.open(name, 'rb')
    else:
        pkl = open(name, 'rb')
    database = pickle.load(pkl, **kwargs)
    pkl.close()
    return database
