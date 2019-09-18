# How to build layer
## signature
    - <funname>(inputs, ..., reuse, name, scope)
    - decorated by core.layer
    - fun = build_fun with fun(inputs) to invoke it
    - return core.run_and_record_fun(fun, name, inputs)
