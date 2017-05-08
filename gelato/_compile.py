from six import exec_
import inspect


def define(name, template, namespace, frame_offset=1):
    try:
        exec_(template, namespace)
    except SyntaxError as e:
        raise SyntaxError(str(e) + ':\n' + template)
    result = namespace[name]
    frm = inspect.stack()[frame_offset]
    mod = inspect.getmodule(frm[0])
    result.__name__ = name
    if mod is None:
        result.__module__ = '__main__'
    else:
        result.__module__ = mod.__name__
    return result
