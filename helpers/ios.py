from sigma import ops

def load(session, checkpoints,
         saver=None,
         var_list=None,
         verbose=True):
    return ops.core.load(session, checkpoints, saver, var_list, verbose)


def save(session, checkpoints,
         saver=None,
         verbose=True,
         **kwargs):
    return ops.core.save(session, checkpoints, saver, verbose, **kwargs)


def import_weights(filename, session,
                   graph=None,
                   collections=[ops.core.Collections.global_variables],
                   verbose=True):
    return ops.core.import_weights(filename,
                                   session,
                                   graph,
                                   collections,
                                   verbose)


def export_weights(filename, session,
                   graph=None,
                   collections=[ops.core.Collections.global_variables],
                   verbose=True):
    return ops.core.export_weights(filename,
                                   session,
                                   graph,
                                   collections,
                                   verbose)


def import_model(filename, session,
                 verbose=True,
                 **kwargs):
    return ops.core.import_model(filename,
                                 session,
                                 verbose,
                                 **kwargs)


def export_model(filename, session,
                 verbose=True,
                 **kwargs):
    return ops.core.export_model(filename,
                                 session,
                                 verbose,
                                 **kwargs)
