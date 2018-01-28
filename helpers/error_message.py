import sys
import traceback

def error_traceback():
    exc_type, exc_value, exc_tb = sys.exc_info()
    return repr(traceback.format_exception(exc_type, exc_value, exc_tb))
