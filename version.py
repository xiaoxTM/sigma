def cvt(version):
    import re
    v = version.split('.')
    if len(v) == 3:
        major, mid, minor = v
        minor = re.match('[0-9]+', minor).group(0)
    elif len(v) == 2:
        major, mid = v
        mid = re.match('[0-9]+', mid).group(0)
        minor = '0'
    else:
        raise ValueError('`version` must be major.mid[.minor], give {}'.format(version))
    return int(major) * 100000 + int(mid) * 100 + int(minor)

def gt(ver1, ver2):
    v1 = cvt(ver1)
    v2 = cvt(ver2)
    return v1 > v2

def ge(ver1, ver2):
    v1 = cvt(ver1)
    v2 = cvt(ver2)
    return v1 >= v2

def eq(ver1, ver2):
    v1 = cvt(ver1)
    v2 = cvt(ver2)
    return v1 == v2

def lt(ver1, ver2):
    v1 = cvt(ver1)
    v2 = cvt(ver2)
    return v1 < v2

def le(ver1, ver2):
    v1 = cvt(ver1)
    v2 = cvt(ver2)
    return v1 <= v2
